# Latent 3D Graph Diffusion


## Unconditional Generation

### Training Topological AE
you can download data and model weights from https://drive.google.com/file/d/1eQOsGfw_XP5S0e1gj0pEJaObU-8i2Y_b/view?usp=sharing
```
cd ./AE_Topology

# get vocabulary for molecular graphs
python get_vocab.py --ncpu 40 < ./AE_topo_weights_and_data/smiles_chembl_mol3d_qm9_drugs.txt > ./AE_topo_weights_and_data/vocab.txt

# preprocess data for more efficient loading
python preprocess.py --train ./AE_topo_weights_and_data/smiles_mol3d_chembl_train.txt --vocab vocab.txt --ncpu 40 --mode single --out_path ./AE_topo_weights_and_data/processed_data_train/
python preprocess.py --train ./AE_topo_weights_and_data/smiles_chembl_mol3d_qm9_drugs.txt --vocab vocab.txt --ncpu 40 --mode single --out_path ./AE_topo_weights_and_data/processed_data/

# train ae
python train_generator_ptl.py  --ddp_num_nodes 1 --ddp_device 1 --train ./AE_topo_weights_and_data/processed_data_train --vocab ./AE_topo_weights_and_data/vocab.txt --save_dir ./AE_topo_weights_and_data/ckpt/pretrained
# if train ae with gssl
python train_generator_gssl_ptl.py  --ddp_num_nodes 1 --ddp_device 1 --train ./AE_topo_weights_and_data/processed_data_train --vocab ./AE_topo_weights_and_data/vocab.txt --save_dir ./AE_topo_weights_and_data/ckpt/pretrained_gssl

# generate smiles to emb dictionary
python generate_embedding.py --train ./AE_topo_weights_and_data/processed_data --vocab ./AE_topo_weights_and_data/vocab.txt --ckpt ./AE_topo_weights_and_data/ckpt/pretrained/last.ckpt --save_fn ./smiles2emb_dict.pt
```


## Conditional Generation on Protein Targets

### Training Topological AE
you can download data and model weights from https://drive.google.com/file/d/1eQOsGfw_XP5S0e1gj0pEJaObU-8i2Y_b/view?usp=sharing
```
cd ./AE_Topology

# get vocabulary for molecular graphs
python get_vocab.py --ncpu 40 < ./AE_topo_weights_and_data/smiles_plus.txt > ./AE_topo_weights_and_data/vocab_pocket_aware.txt

# preprocess data for more efficient loading
python preprocess.py --train ./AE_topo_weights_and_data/smiles_mol3d_chembl_train.txt --vocab vocab_pocket_aware.txt --ncpu 40 --mode single --out_path ./AE_topo_weights_and_data/processed_data_pocket_train/
python preprocess.py --train ./AE_topo_weights_and_data/smiles_plus.txt --vocab vocab_pocket_aware.txt --ncpu 40 --mode single --out_path ./AE_topo_weights_and_data/processed_data_pocket/

# train ae
python train_generator_ptl.py  --ddp_num_nodes 1 --ddp_device 1 --train ./AE_topo_weights_and_data/processed_data_pocket_train --vocab ./AE_topo_weights_and_data/vocab_pocket_aware.txt --save_dir ./AE_topo_weights_and_data/ckpt/pocket_pretrained
# if train ae with gssl
python train_generator_gssl_ptl.py  --ddp_num_nodes 1 --ddp_device 1 --train ./AE_topo_weights_and_data/processed_data_pocket_train --vocab ./AE_topo_weights_and_data/vocab_pocket_aware.txt --save_dir ./AE_topo_weights_and_data/ckpt/pocket_pretrained_gssl

# generate smiles to emb dictionary
python generate_embedding.py --train ./AE_topo_weights_and_data/processed_data_pocket --vocab ./AE_topo_weights_and_data/vocab_pocket_aware.txt --ckpt ./AE_topo_weights_and_data/ckpt/pocket_pretrained/last.ckpt --save_fn ./smiles2emb_dict_pocket.pt
```

### Training Geometric AE
you can download model weights and samples from https://drive.google.com/file/d/1Rzzoi7iBBrLuoa0-sCEhUSYrWXutue5M/view?usp=sharing

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

### Sampling and evaluating
```
cd ./AE_Geometry_and_Conditional_Latent_Diffusion

# sample latent embeddings
python -m scripts.sample_z configs/training.yml

# reconstruct 2d
python -m scripts.sample_2d

# reconstruct 3d
python -m scripts.sample_3d configs/sampling.yml

# evaluate
python -m scripts.evaluate_diffusion outputs --docking_mode vina_score --protein_root data/test_set --data_id $data_id
```

