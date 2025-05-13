### Learning the cell gut microbiome dataset

Paper of the original dataset: https://www.cell.com/cell/fulltext/S0092-8674(24)01430-2?dgcid=raven_jbs_aip_email

Box folder of the project: https://uofi.app.box.com/folder/311808882946?s=n64i84joa37r52fmfb2r4098nbn84niq

VAE models: 

- A 2-branched VAE model for reconstructing the (preprocessed) relative abundance of species in each sample. 
- A scVI-styled VAE model for reconstructing the absolute abundance of species in each sample. 

Technically, both structures can be used to do binary->non-binary reconstruction (learning ecology insights; doing perturbation experiments etc). We expect the scVI-styled VAE can be trained to denoise the data and give more solid data for the b2nb learnings. 