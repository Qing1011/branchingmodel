/*********************************************************************
 * branching.cpp
 * Keep in mind:
 * <> Use 0-based indexing as always in C or C++
 * <> Indexing is column-based as in Matlab (not row-based as in C)
 * <> Use linear indexing.  [i-1+(j-1)*num_row] in C++ instead of [i][j] in Matlab (note starting index)
 * Adapted from the code by Shawn Lankton (http://www.shawnlankton.com/2008/03/getting-started-with-mex-a-short-tutorial/)
 ********************************************************************/
// #include <matrix.h>
// #include <mex.h>
#include <pybind11/pybind.h> 

namespace py = pybind11;

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm> 
using namespace std; 

/* Definitions to keep compatibility with earlier versions of ML */
#ifndef MWSIZE_MAX
typedef int mwSize;
typedef int mwIndex;
typedef int mwSignedIndex;

#if (defined(_LP64) || defined(_WIN64)) && !defined(MX_COMPAT_32)
/* Currently 2^48 based on hardware limitations */
# define MWSIZE_MAX    281474976710655UL
# define MWINDEX_MAX   281474976710655UL
# define MWSINDEX_MAX  281474976710655L
# define MWSINDEX_MIN -281474976710655L
#else
# define MWSIZE_MAX    2147483647UL
# define MWINDEX_MAX   2147483647UL
# define MWSINDEX_MAX  2147483647L
# define MWSINDEX_MIN -2147483647L
#endif
#define MWSIZE_MIN    0UL
#define MWINDEX_MIN   0UL
#endif

void mexFunction(int nlmxhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
//declare variables
    mxArray *mxnl, *mxpart, *mxw, *mxT0, *mxNewInf, *mxpara, *mxpop;//input
    mxArray *mxNewInf1;//output
    const mwSize *dims;
    double *nl, *part, *w, *T0, *NewInf, *para, *pop;//input
    double *NewInf1;//output
    int num_mp, num_loc, T, Tcnt;

//associate inputs
    mxnl = mxDuplicateArray(prhs[0]);
    mxpart = mxDuplicateArray(prhs[1]);
    mxw = mxDuplicateArray(prhs[2]);
    mxT0 = mxDuplicateArray(prhs[3]);
    mxNewInf = mxDuplicateArray(prhs[4]);
    mxpara = mxDuplicateArray(prhs[5]);
    mxpop = mxDuplicateArray(prhs[6]);
    
//figure out dimensions
    dims = mxGetDimensions(prhs[0]);//number of subpopulation
    num_mp = (int)dims[0];
    dims = mxGetDimensions(prhs[1]);//number of locations
    num_loc = (int)dims[0]-1;
    dims = mxGetDimensions(prhs[4]);//simulation period
    T = (int)dims[1];
    
//associate outputs
    mxNewInf1 = plhs[0] = mxCreateDoubleMatrix(num_loc,T,mxREAL);
    
//associate pointers
    nl = mxGetPr(mxnl);
    part = mxGetPr(mxpart);
    w = mxGetPr(mxw);
    T0 = mxGetPr(mxT0);
    NewInf = mxGetPr(mxNewInf);
    para = mxGetPr(mxpara);
    pop = mxGetPr(mxpop);
    
    NewInf1 = mxGetPr(mxNewInf1);
    
    ////////////////////////////////////////
    //do something
    default_random_engine generator((unsigned)time(NULL));
//     clock_t start,finish;
// 	double totaltime;
//     finish=clock();
//     totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
//     cout<<"Elapsed time 2: "<<totaltime<<" seconds."<<endl;
//     start=clock();
    //initialize auxillary variables
    int i,j,l,t;
    int newinfection;//number of infectors
    int loc;//location of infector
    double v,temp;
    double totalinfection[num_loc];
    
    /////////////////////////////
    //change index in nl and part (0-based index)
    for (i=0; i<num_mp; i++)
        nl[i]=nl[i]-1;
    for (i=0; i<num_loc+1; i++)
        part[i]=part[i]-1;
    Tcnt=T0[0]-1;//the current time (note index from 0)
    //total infection
    for (l=0; l<num_loc; l++){
        totalinfection[l]=0;
        for (t=0;t<=Tcnt;t++)
            totalinfection[l]=totalinfection[l]+NewInf[l+t*num_loc];
    }
    //initialize NewInf1,LocalInf1
    for (l=0; l<num_loc; l++){
        for (t=0;t<T;t++){
            NewInf1[l+t*num_loc]=NewInf[l+t*num_loc];
        }
    }
    //prepare generators
    //para: R0,r,Z;Zb;D;Db,alpha
    double R0=para[0],r=para[1];
    double p=r/(R0+r);
    //////use a discrete distribution to generate NB
    //prepare NB pdf
    vector<double> weights;
    
    for (i=0; i<100; i++){
        long double temp1=tgamma((long double)(r+i))/tgamma((long double)r)/tgamma((long double)(i+1))*pow(p,r)*pow(1-p,i);
        weights.push_back(temp1);
    }
    
    std::discrete_distribution<> secondary(weights.begin(), weights.end());
    std::random_device rd;
    
    double Za=para[2],Zb=para[3];
    std::gamma_distribution<double> Zrnds(Za,Zb);
    
    double Da=para[4],Db=para[5];
    std::gamma_distribution<double> Drnds(Da,Db);
    
    //define variables
    int z;//secondary infection
    double Z;//latency
    double D;//Infectious
    int tdist;//time of secondary infection

    //loop through all locations
    for (l=0; l<num_loc; l++){
        newinfection = NewInf[l+Tcnt*num_loc];
        if (newinfection>0){
            //for each infector
            for (i=0; i<newinfection; i++){
                
                //generate secondary infections,negative binomial             
                z=secondary(rd);

                if (z>0){
                    //decide where is the infector
                    v=(double)(rand())/RAND_MAX;
                    temp=0;
                    for (j=part[l]; j<part[l+1]; j++){
                        temp=temp+w[j];
                        if (v<=temp){
                            loc=nl[j];//current location
                            break;
                        }   
                    }

                    //consider population immunity
                    z=round(z*(1-totalinfection[loc]/pop[loc]));

                    //distribute over time
                    for (j=0; j<z; j++){
                        
                        Z=Zrnds(generator);
                        D=Drnds(generator);
                        
                        v=(double)(rand())/RAND_MAX;
                        
                        tdist=ceil(Z+v*D);//uniform distribute
                        
                        //add to new infection
                        if (Tcnt+tdist<T){
                            NewInf1[loc+(Tcnt+tdist)*num_loc]=NewInf1[loc+(Tcnt+tdist)*num_loc]+1;
                        }
                    }
                    
                }

            }
        }
    }

    /////////////////////
    return;
}

PYBIND11_MODULE(branching, handle){
    handle/doc() = "branching";
    handle.def("branching", &mexFunction);}
