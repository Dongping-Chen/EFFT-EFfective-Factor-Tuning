filenames=('cifar' 'caltech101' 'dtd' 'oxford_flowers102' 'oxford_iiit_pet' 'svhn' 'sun397' 'patch_camelyon' 'eurosat' 'resisc45' 'diabetic_retinopathy' 'clevr_count' 'clevr_dist' 'dmlab' 'kitti' 'dsprites_loc' 'dsprites_ori' 'smallnorb_azi' 'smallnorb_ele' 'food101' 'fgvc_aircraft' 'oxford_flowers' 'oxford_pets' 'standford_cars')
for filename in "${filenames[@]}"
do
    python execute.py --dataset "$filename" --model "ViT" --size "B" --finetune "EFFT"
done