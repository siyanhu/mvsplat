# python -m src.main +experiment=7s_scene_stairs_crossvalid \
# checkpointing.load=checkpoints/re10k.ckpt \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=datasets_crossvalid/scene_stairs/2024_09_08/test/evaluation_index_7s_scene_stairs.json \
# test.compute_scores=true

# python -m src.main +experiment=7s_scene_fire_crossvalid \
# checkpointing.load=checkpoints/re10k.ckpt \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=datasets_crossvalid/scene_fire/2024_09_08/test/evaluation_index_7s_scene_fire.json \
# test.compute_scores=true

# python -m src.main +experiment=re10k \
# data_loader.train.batch_size=14 \
# output_dir='/home/siyanhu/Gits/mvsplat/outputs_train' \
# mode=train

python -m src.main +experiment=7s_scene_stairs_train \
data_loader.train.batch_size=14 \
mode=train \
# output_dir='/home/siyanhu/Gits/mvsplat/outputs_train'