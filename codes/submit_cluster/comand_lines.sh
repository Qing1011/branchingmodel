# !/bin/bash
for j in 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6.0 6.1 6.2 6.3 6make
    do
    for i in 20.0 10.0 2.0 1.0 0.5 0.2 0.1 0.05 0.025; 
    do 
    mkdir "branching_R0-${j}_r-${i}"; 
    done
done

for j in 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2; 
do 
mkdir "branching_R0-${j}_r-0.1"; 
done

for i in 96 105 114 123 132 141 150 159;
do
   qsub branching_${i}.pbs;
done

for i in 12.5 4.0 0.4 0.08;
do 
mkdir "branching_R0-1.5_r-${i}"; 
done

for i in 20.0 10.0 2.0 1.0 0.5 0.2 0.1 0.05 0.025 7.4 5.0 2.5 13.333 3.333 1.333 0.667 0.37 0.286 0.133 0.067 0.033;
do 
mkdir "branching_R0-5.5_r-${i}"; 
done

for i in 0 1 2 3 4 5 6 7 8 9;
do
   qsub branching_${i}.pbs;
done



for i in 1.0 0.5 0.2 0.1 0.05 5.0 2.5 3.333 1.333 0.667 0.37 0.286 0.133 0.067 0.033; 
do
  tar -czvf branching_R0-5.5_r-${i}.tar.gz branching_R0-5.5_r-${i}/
done


for i in 20.0 10.0 2.0 1.0 0.5 0.05 0.025 7.4 5.0 2.5 13.333 3.333 1.333 0.667 0.37 0.286 0.133 0.067 0.033; 
do
mv branching_R0-5.5_r-${i}/ python_cutoff_addno/
done


  # Unzip the file
#   tar -xzvf branching_R0-5.5_r-${i}.tar.gz
mv branching_R0-5.5_r-${i}/ python_cutoff_addno/
  # Move the unzipped directory to target_directory

