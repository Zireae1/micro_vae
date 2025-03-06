# micro_vae contents:

**data/** - folder with data tables (note: metadata has column "uid" that corresponds to row index in the abundance data table)
health.noab - healthy patients with no antibiotic administration
health.ab - healthy people who took antibiotics recently
disease - people from various disease cohorts

**VAE_b2b.ipynb** - train model that reconstructs binary to binary data (data binarization included)

**VAE_b2nb.ipynb** - train model that reconstructs binary to non binary renormalized abundance data (data binarization included)

**VAE_nb2nb.ipynb** - train model that reconstructs non binary renormalized abundance to non binary renormalized abundance data (data binarization included)
