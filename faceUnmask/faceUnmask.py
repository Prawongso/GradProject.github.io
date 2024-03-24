import matplotlib.image as mpimg
from utils.configuration import Configuration
from utils.architectures import UNet
from utils.model import Mask2FaceModel
import matplotlib.pyplot as plt
import cv2

configuration = Configuration()
filters = (64, 128, 128, 256, 256, 512)
kernels = (7, 7, 7, 3, 3, 3)
input_image_size = (256, 256, 3)
architecture = UNet.RESNET
model = Mask2FaceModel.build_model(architecture=architecture, input_size=input_image_size, filters=filters,
                                   kernels=kernels, configuration=configuration)
model.built = True
model.load_weights('faceUnmask\models\model.h5')
# Load a single JPEG image
input_img_path = "faceUnmask\input\VincentM.jpg"
# Preprocess the input image
# (You may need to resize or apply other preprocessing steps depending on your model)
# Example:
# input_img = preprocess_input(input_img)

# Generate the output using the model
generated_output = model.predict('faceUnmask\input\VincentM.jpg')

# Display the input image and the generated output
f, axarr = plt.subplots(1, 1)
axarr.imshow(generated_output)
axarr.set_title('Generated Output')
axarr.axis('off')
f.savefig('output.jpg', format='jpg', dpi=300)
plt.show()
