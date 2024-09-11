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

# python -m src.main +experiment=dtu \
# checkpointing.load=checkpoints/acid.ckpt \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=assets/evaluation_index_dtu_nctx3.json \
# dataset.view_sampler.num_context_views=3 \
# test.compute_scores=true

# python -m src.main +experiment=7s_scene_stairs_crossvalid \
# checkpointing.load=checkpoints/re10k.ckpt \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=datasets__crossvalid_preset/7s/scene_stairs/test/evaluation.json \
# dataset.view_sampler.num_context_views=10 \
# test.compute_scores=true

# python -m src.main +experiment=7s_scene_stairs_crossvalid \
# checkpointing.load=checkpoints/re10k.ckpt \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=datasets__crossvalid_preset/7s/scene_stairs/test/evaluation.json \
# dataset.view_sampler.num_context_views=10 \
# test.compute_scores=true

# python -m src.main +experiment=re10k \
# checkpointing.load=checkpoints/acid.ckpt \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
# dataset.view_sampler.num_context_views=2 \
# test.compute_scores=true

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
# checkpointing.load=checkpoints/re10k.ckpt \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=datasets__crossvalid_preset_n10/camb/scene_KingsCollege/test/evaluation.json \
# dataset.view_sampler.num_context_views=10 \
# test.compute_scores=true

# python -m src.main +experiment=camb_scene_KingsCollege_n2.yaml \
# checkpointing.load=checkpoints/re10k.ckpt \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=datasets/camb/n2/scene_KingsCollege/test/evaluation.json \
# dataset.view_sampler.num_context_views=2 \
# test.compute_scores=true

# python -m src.main +experiment=7s_scene_stairs_n2.yaml \
# checkpointing.load=checkpoints/re10k.ckpt \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=datasets/7s/n2/scene_stairs/test/evaluation.json \
# dataset.view_sampler.num_context_views=2 \
# test.compute_scores=true

# python -m src.main +experiment=7s_scene_fire_n2.yaml \
# checkpointing.load=checkpoints/re10k.ckpt \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=datasets/7s/n2/scene_fire/test/evaluation.json \
# dataset.view_sampler.num_context_views=5 \
# test.compute_scores=true

python src/scripts/convert_7scamb_hs5.py

bash datasets/7s/n2/scene_stairs/test/command.sh 
bash datasets/7s/n2/scene_fire/test/command.sh 
bash datasets/camb/n2/scene_KingsCollege/test/command.sh 

bash datasets/7s/n5/scene_stairs/test/command.sh 
bash datasets/7s/n5/scene_fire/test/command.sh 
bash datasets/camb/n5/scene_KingsCollege/test/command.sh 

bash datasets/7s/n10/scene_stairs/test/command.sh 
bash datasets/7s/n10/scene_fire/test/command.sh 
bash datasets/camb/n10/scene_KingsCollege/test/command.sh 