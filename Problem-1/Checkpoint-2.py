
# coding: utf-8

# In[38]:


import pandas as pd
from os.path import join


# In[39]:


data_path = "../../dataset_works.csv"#join("..", "..", "Dataset-1", "selfie_dataset.txt")
image_path = "../../dataset/"#join("..", "..", "Dataset-1", "selfie_dataset.txt")#join("..", "..", "Dataset-1", "images")


# In[40]:


headers = [
    "image_name", "score", "partial_faces" ,"is_female" ,"baby" ,"child" ,"teenager" ,"youth" ,"middle_age" ,"senior" ,"white" ,"black" ,"asian" ,"oval_face" ,"round_face" ,"heart_face" ,"smiling" ,"mouth_open" ,"frowning" ,"wearing_glasses" ,"wearing_sunglasses" ,"wearing_lipstick" ,"tongue_out" ,"duck_face" ,"black_hair" ,"blond_hair" ,"brown_hair" ,"red_hair" ,"curly_hair" ,"straight_hair" ,"braid_hair" ,"showing_cellphone" ,"using_earphone" ,"using_mirror", "braces" ,"wearing_hat" ,"harsh_lighting", "dim_lighting"
]
df_image_details = pd.read_csv(data_path, names=headers, delimiter="\t")
df_image_details.head(5)


# In[41]:


df_image_details = df_image_details.sample(frac=1)


# In[42]:


needed_columns = [
    'image_name',
    'is_female',
    'baby',
    'child',
    'teenager',
    'youth',
    'middle_age',
    'senior'
]


# In[43]:


df_image_details = df_image_details[needed_columns]
df_image_details = df_image_details[df_image_details.is_female != 0]
df_image_details.replace(to_replace=-1, value=0, inplace=True)


# In[44]:


df_image_details.head(5)


# In[45]:


image_names = df_image_details.image_name.values.copy()
image_attrs = df_image_details[needed_columns[1:]].values.copy()


# In[46]:


image_paths = [join(image_path, iname) + '.jpg' for iname in image_names]


# In[48]:


image_paths_train, image_paths_test = image_paths[:-250], image_paths[-250:]
image_attrs_train, image_attrs_test = image_attrs[:-250], image_attrs[-250:]


# In[49]:


from keras.utils import Sequence
import numpy as np
import cv2


# In[50]:


cv2.__version__


# In[51]:


class ImageGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # read your data here using the batch lists, batch_x and batch_y
        x = [self.read_image(filename) for filename in batch_x] 
        y = [atrributes for atrributes in batch_y]
        return np.array(x), np.array(y)
    
    def read_image(self, fname):
        im = cv2.imread(fname)
        im = cv2.resize(im, (224, 224))
        return im/255.


# # Training Model

# In[52]:


from keras.applications import resnet50
from keras.layers import Dense, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


# In[53]:


model_rnet = load_model("resnet50.hdf5")


# In[54]:


for layer in model_rnet.layers:
    layer.trainable = False


# In[56]:


model_output = Dense(1024, activation='relu', )(model_rnet.get_layer('avg_pool').output)
model_output = Dropout(0.5)(model_output)
model_output = Dense(512, activation='relu')(model_output)
model_output = Dropout(0.5)(model_output)
# model_output = Dense(512, activation='relu')(model_output)
# model_output = Dropout(0.33)(model_output)
# model_output = Dense(128, activation='relu')(model_output)
# model_output = Dropout(0.2)(model_output)
model_output = Dense(7, activation='sigmoid')(model_output)


# In[57]:


model = Model(inputs=[model_rnet.input], outputs=[model_output])


# In[58]:


model.summary()


# In[59]:


model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])


# In[60]:


train_gen = ImageGenerator(image_paths_train, image_attrs_train, batch_size=32)
test_gen = ImageGenerator(image_paths_test, image_attrs_test, batch_size=32)


# In[61]:


train_len = len(image_paths_train)
test_len = len(image_paths_test)
train_len, test_len


# In[62]:


tot = 4520 + 1025 + 796 + 1474 + 2380 + 755 + 677
class_weights = {
    0: 4520/tot , 1: 1025/tot , 2: 796/tot , 3: 1474/tot , 4: 2380/tot , 5: 755/tot , 6: 677/tot
}

# class_weights = [i/tot for i in class_weights]


# In[ ]:


model.fit_generator(train_gen, validation_data=test_gen, epochs=200, 
                    steps_per_epoch=train_len // 32,
                   validation_steps=10, use_multiprocessing=False, 
                   callbacks=[
                       ReduceLROnPlateau(patience=2, verbose=1),
                       ModelCheckpoint('chpt-2-new-d.hdf5', verbose=1, save_best_only=True)
                   ])

