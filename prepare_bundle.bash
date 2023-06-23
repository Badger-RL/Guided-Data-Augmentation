cd ..

zip -r bundle.zip GuidedDataAugmentationForRobotics -x GuidedDataAugmentationForRobotics/results/**\* -x GuidedDataAugmentationForRobotics/env/**\* -x GuidedDataAugmentationForRobotics/bundle.zip -x GuidedDataAugmentationForRobotics/.git/**\* -x GuidedDataAugmentationForRobotics/credentials.json -x GuidedDataAugmentationForRobotics/src/wandb/**\* -x GuidedDataAugmentationForRobotics/src/logdata/**\* -x GuidedDataAugmentationForRobotics/src/policy/**\* 
mv bundle.zip GuidedDataAugmentationForRobotics
