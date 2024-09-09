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

# python -m src.main +experiment=7s_scene_stairs_train \
# data_loader.train.batch_size=14 \
# mode=train \
# # output_dir='/home/siyanhu/Gits/mvsplat/outputs_train'

python -m src.main +experiment=dtu \
checkpointing.load=checkpoints/acid.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_dtu_nctx3.json \
dataset.view_sampler.num_context_views=3 \
test.compute_scores=true

python -m src.main +experiment=7s_scene_stairs_crossvalid \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=datasets__crossvalid_preset/7s/scene_stairs/test/evaluation.json \
dataset.view_sampler.num_context_views=10 \
test.compute_scores=true

python -m src.main +experiment=7s_scene_stairs_crossvalid \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=datasets__crossvalid_preset/7s/scene_stairs/test/evaluation.json \
dataset.view_sampler.num_context_views=10 \
test.compute_scores=true

# python -m src.main +experiment=7s_scene_fire_crossvalid \
# checkpointing.load=checkpoints/re10k.ckpt \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=datasets_crossvalid_evenly/scene_fire/2024_09_09/test/evaluation_index_7s_scene_fire.json \
# dataset.view_sampler.num_context_views=2 \
# test.compute_scores=true

# python -m src.main +experiment=camb_scene_KingsCollege_crossvalid \
# checkpointing.load=checkpoints/re10k.ckpt \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=datasets__crossvalid_preset/camb/scene_KingsCollege/test/evaluation.json \
# test.compute_scores=true

# python -m src.main +experiment=camb_scene_KingsCollege_crossvalid \
# checkpointing.load=checkpoints/acid.ckpt \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=datasets__crossvalid_preset/camb/scene_KingsCollege/test/evaluation.json \
# test.compute_scores=true