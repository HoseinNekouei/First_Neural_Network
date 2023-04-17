import cv2
import numpy as np
import glob
from keras.models import load_model


output_label = ['fire', 'non-fire']
clf = load_model('neural_network\\mlp.h5')

for item in glob.glob('fire_detection\\test_image\\*'):
    img = cv2.imread(item)
    r_img = cv2.resize(img, (32, 32))
    r_img = r_img / 255.0
    r_img = np.array([r_img])

    pred_output = clf.predict(r_img)[0]
    max_pred = np.argmax(pred_output)
    output = output_label[max_pred]

    if output == 'fire':
        cv2.putText(img, 'Fire !!!', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 2)
    else:
        cv2.putText(img, 'non_Fire', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 0), 2)

    cv2.imshow('image', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
