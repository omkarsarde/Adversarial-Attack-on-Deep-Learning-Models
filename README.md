# Adversarial Attack on Deep Learning Models
 Demonstration of Importance against securing Computer Vision models against adversarial attacks.
 <br>
 Try predicting the class of the Adv_1 image through any State of the Art model while using the following transformation for the image:
 <br>
 ```python
 def transform_input(image):
    """
    Transform image to model requirements
    """
    img = Image.open(image)
    resize = transforms.Compose([transforms.Resize(244), transforms.ToTensor(),
                                 transforms.Normalize(
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]
                                 )])
    img = resize(img)
    batch_transformed = torch.unsqueeze(img, 0)
    return batch_transformed
 ```
 <br>
 Even without using the transformation the RAW ADVERSARIAL INPUT will have less probability of predictions compared to the orginal peppers image.
 <br>
 
 
 # IMAGES:
 <table>
  <tr>
    <td>ORIGINAL IMAGE</td>
     <td>ADVERSARIAL IMAGE</td>
  </tr>
  <tr>
    <td valign="top"><img src="/peppers.jpg",width="50%"></td>
    <td valign="top"><img src="/Adv_1.jpg",width="50%"></td>
  </tr>
 </table>
 
 
 # Approach
 The Approach is in two parts. First a white box attack is launched on a vgg16 model trained on imagenet to generate
