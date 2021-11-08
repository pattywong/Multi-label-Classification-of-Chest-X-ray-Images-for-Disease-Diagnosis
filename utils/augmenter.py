from imgaug import augmenters as iaa

def augmenter_setting(config):
    augmenter = iaa.Sequential(config, random_order=True,)
    return augmenter

def create_augmenter(augmented_prob,list_of_augment_operation,random_order =True):
    augmenter = iaa.Sequential(list_of_augment_operation, random_order=random_order,)
    augmenter = iaa.Sometimes(augmented_prob,augmenter)
    return augmenter

config = [
         iaa.Fliplr(0.5),
         iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))),
         iaa.ContrastNormalization((0.75, 1.5)),
         iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
         iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-8, 8),
            shear=(-3, 3))
        ]

augmenter = create_augmenter(1.00,config)