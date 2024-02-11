# Latent_3D_Graph_Diffusion

### Training Topological AE
download data and model weights from https://drive.google.com/file/d/1eQOsGfw_XP5S0e1gj0pEJaObU-8i2Y_b/view?usp=sharing
```
cd ./AE_Topology

# get vocabulary for molecular graphs
python get_vocab.py --ncpu 40 < ./AE_topo_weights_and_data/smiles_plus.txt > ./AE_topo_weights_and_data/vocab_pocket_aware.txt

# preprocess data for more efficient loading
python preprocess.py --train ./AE_topo_weights_and_data/smiles_mol3d_chembl_train.txt --vocab vocab_pocket_aware.txt --ncpu 40 --mode single --out_path ./AE_topo_weights_and_data/processed_data_pocket_train/

# train ae
python train_generator_ptl.py  --ddp_num_nodes 1 --ddp_device 1 --train ./AE_topo_weights_and_data/processed_data_pocket_train --vocab ./AE_topo_weights_and_data/vocab_pocket_aware.txt --save_dir ./AE_topo_weights_and_data/ckpt/pocket_pretrained
```

### Training Geometric AE
download data and model weights from https://drive.google.com/file/d/1Rzzoi7iBBrLuoa0-sCEhUSYrWXutue5M/view?usp=sharing

download data following https://github.com/guanjq/targetdiff#data
```
cd ./AE_Geometry_and_Conditional_Latent_Diffusion

# train ae
python -m scripts.train_ae configs/training.yml

# generate 2d and 3d embeddings
python -m scripts.generate_embedding configs/sampling.yml
```

### Training Diffusion Model
```
cd ./AE_Geometry_and_Conditional_Latent_Diffusion

python -m scripts.train_latent_diffusion configs/training.yml
```

### Sampling
```
cd ./AE_Geometry_and_Conditional_Latent_Diffusion

# sample latent embeddings
python -m scripts.sample_z configs/training.yml

# reconstruct molecular graphs
python -m scripts.sample_2d

# reconstruct molecular conformer
python -m scripts.sample_3d configs/sampling.yml
```

