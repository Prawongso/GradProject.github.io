import sys         
sys.path.insert(1,'D:/School stuff/Github/GradProject.github.io/faceUnmask')   

import matplotlib.pyplot as plt
from utils.configuration import Configuration
from utils.architectures import UNet
from utils.model import Mask2FaceModel

configuration = Configuration()

filters = (64, 128, 128, 256, 256, 512)
kernels = (7, 7, 7, 3, 3, 3)
input_image_size = (256, 256, 3)
architecture = UNet.RESNET
training_epochs = 20
batch_size = 12

model = Mask2FaceModel.build_model(architecture=architecture, input_size=input_image_size, filters=filters,
                                  kernels=kernels, configuration=configuration)
#model.built = True
model = Mask2FaceModel.load_model('faceUnmask\models\model.keras')
# Load a single JPEG image
input_img_path = "faceUnmask\input\input1.jpg"
# Preprocess the input image
# (You may need to resize or apply other preprocessing steps depending on your model)
# Example:
# input_img = preprocess_input(input_img)

# Generate the output using the model
generated_output = model.predict('faceUnmask\input\VincentM.jpg')

# Display the input image and the generated output
f, axarr = plt.subplots(1, 1)
axarr.imshow(generated_output)
axarr.set_title('AI Generated Output')
axarr.axis('off')
f.savefig('FaceRec/unknown_image/Unknown3.jpg', format='jpg', dpi=300)
plt.show()
plt.waitforbuttonpress()

# Check if the key pressed is Enter (keycode 13)
if plt.get_current_fig_manager().toolbar.mode == '':
    plt.close(f)
    print("Loading face recognition...") 
    exec(open("FaceRec/FaceRec.py").read())  