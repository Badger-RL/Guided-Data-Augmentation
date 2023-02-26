cd ..

zip -r bundle.zip GuidedDataAugmentationForRobotics -x GuidedDataAugmentationForRobotics/env/**\* -x GuidedDataAugmentationForRobotics/bundle.zip -x GuidedDataAugmentationForRobotics/.git/**\* -x GuidedDataAugmentationForRobotics/credentials.json -x GuidedDataAugmentationForRobotics/src/wandb/**\* -x GuidedDataAugmentationForRobotics/src/policy/**\* -x GuidedDataAugmentationForRobotics/src/datasets/expert/translate_robot_and_ball/100000_4.hdf5 -x GuidedDataAugmentationForRobotics/src/datasets/random/translate_robot_and_ball/100000_4.hdf5
mv bundle.zip GuidedDataAugmentationForRobotics