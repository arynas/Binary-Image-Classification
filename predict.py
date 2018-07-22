import numpy as np
from keras.preprocessing import image
from keras.models import model_from_yaml

test_image = image.load_img('data/test_set/cats/cat.4714.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#load model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
model.load_weights("model.h5")

result = model.predict(test_image)


# training_set.class_indices
print(result)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print("The prediction is", prediction)
