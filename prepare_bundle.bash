cd ..

zip -r bundle.zip GuidedDataAugmentationForRobotics -x GuidedDataAugmentationForRobotics/env/**\* -x GuidedDataAugmentationForRobotics/bundle.zip -x GuidedDataAugmentationForRobotics/.git/**\* -x GuidedDataAugmentationForRobotics/credentials.json
mv bundle.zip GuidedDataAugmentationForRobotics