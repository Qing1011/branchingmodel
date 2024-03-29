# Superspreading shapes the early spatial spread of emerging infectious diseases

Our study examines the impact of superspreading and its inherent stochasticity on the initial spatial spread of novel pathogens. 

## Getting Started
A python3.x and Notebook environment is required to run the code.
### Python Packages
**numpy**
**pandas**
**matplotlib**
**scipy**
**sklearn**
**tensorflow**
**networkx**
**gzip**
**torch**
**torch_geometric**

### Commuting Data
The cleaned weighted commuting matrix is available in the `data` folder called W_avg.csv. The original commuting data is available at https://www.census.gov/data/datasets/2017/demo/metro-micro.html. The population data is saved as pop_new.csv. 

## Branching model  
We introduce a branching process model incorporating a spatial layout of US counties and inter-county commuting to simulate invasion dynamics of pathogens with superspreading. To represent transmission heterogeneity, we model secondary infections generated by an infectious person using a negative binomial distribution with a mean  and a dispersion parameter.

We tested the seeding of a range of dispersion rates and choose $100$ as it is about the smallest seed number which can spread out for all the dispersion rate we choose. We do not upload the simulation results as it is too large, but we do provide the codes to run the simulations and the notebook to reproduce the figures in the paper.

## Metapopulation model with stochasticity
Compared with metapopulation simulations, the branching process model with strong superspreading produces a faster spatial invasion during outbreak onset but a slower spatial progression subsequently.

We coded the metapoluation model first and then add the stochasticity to the model. The codes are in the `codes` folder. The notebook to reproduce the figures in the paper is also in the `notebook` folder.

To create the map of the US, we use the **county shapefile** from https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.2020.html#list-tab-1883739534. We used the 1 : 500,000 (national) resolution. 

## Inferring the superspreading parameter dispersion rate

### Likelihood-based approach
We find that, by leveraging the spatial spread pattern, a likelihood-based approach can identify dispersion parameters within a certain range.
### Deep learning approach
We further demonstrate that deep learning can distinguish between different levels of superspreading using spatial spread. Our study offers a new approach to quantifying the superspreading potentials of pathogens using more accessible population-level observations.

We do not upload all the training/test results.We provide the codes to create the training set and to select the models and the notebook to reproduce the figures in the paper. 
### Movies
We create movies to show the spatial spread of the simulated outbreaks. The data for movies should be simulated and the creation of the figures is in the notebook. The GIF figures can be found in https://figshare.com/s/7d0a438307d0a7c4f6b1.
