import torch
import glob
from torchvision.transforms import ToPILImage
import matplotlib
import torch.nn
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
import requests, io
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split

"""
Fooling Neural Networks: Making State of the Art models misclassify inputs
We create FGSM based black box attacks on Resent50, making it completely missclassify the input image.
Black box Adversarial Attacks created using white box attack on VGG model using Fast Gradient Sign Method
"""

class Data(Dataset):
    """
    Elementary dataset implementation from pytorch, nothing fancy
    """
    def __init__(self, root_dir, transform=None):
        """
        Initialization
        """
        self.transform = transform
        self.root_dir = root_dir
        self.labels, self.samples = [], []
        self._init_dataset()

    def __len__(self):
        """
        Length of the DS
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get an item(sample) using id
        """
        x = self.samples[idx]
        if self.transform:
            x = self.transform(x)
        y = self.labels[idx]
        return x, y

    def _init_dataset(self):
        """
        Read the directory
        """
        for file in glob.iglob(self.root_dir + "/*.jpg"):
            self.labels.append(812)
            img = Image.open(file)
            img = img.convert('RGB')
            self.samples.append(img)


def transform_adv(image):
    """
    Transform image to model requirements
    """
    img = Image.open(image)
    resize = transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                 transforms.RandomResizedCrop(size=244, ratio=(0.8, 0.8)),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.RandomRotation(degrees=30),
                                 transforms.Resize(244), transforms.ToTensor(),
                                 transforms.Normalize(
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]
                                 )])
    img = resize(img)
    batch_transformed = torch.unsqueeze(img, 0)
    return batch_transformed


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


def get_label(model, image, p1):
    """
    Get Predicted Label by model as models are trained on ImageNet and have 1000 label classes
    """
    model.eval()
    if p1 == True:
        image = transform_input(image)
    else:
        image = transform_adv(image)
    image = Variable(image, requires_grad=True)
    output = model.forward(image)
    label_idx = torch.max(output.data, 1)[1][0].item()
    print(label_idx)
    labels_link = "https://savan77.github.io/blog/labels.json"
    labels_json = requests.get(labels_link).json()
    labels = {int(idx): label for idx, label in labels_json.items()}
    x_pred = labels[label_idx]
    print(x_pred)
    output_probs = F.softmax(output, dim=1)
    x_pred_prob = round((torch.max(output_probs.data, 1)[0][0].item()) * 100, 4)
    print(x_pred_prob)
    return label_idx, x_pred, x_pred_prob, labels


def gen_adv(images, targets, model, outnames):
    """
    Stage the FGSM white box attack on first model to generate black box Adversarial Examples on the second model
    """
    imagetensor, variabledata, grads, epsilons, xpreds, xadvpred, xpredprob, xadvpredprob = [], [], [], [], [], [], [], []

    # Similar to fundamental training loops of any pytorch training function
    for i in range(len(images)):
        label_idx, x_pred, x_pred_prob, labels = get_label(model, images[i], True)
        model.eval()
        y_target = Variable(torch.LongTensor([targets[i]]), requires_grad=False)
        epsilon = 0.25
        num_steps = 5
        alpha = 0.025
        image_tensor = transform_input(images[i])
        img_variable = Variable(image_tensor, requires_grad=True)

        total_grad = 0
        for i in range(num_steps):
            zero_gradients(img_variable)
            output = model.forward(img_variable)
            loss = torch.nn.CrossEntropyLoss()
            loss_cal = loss(output, y_target)
            loss_cal.backward()
            x_grad = alpha * torch.sign(img_variable.grad.data)
            adv_temp = img_variable.data - x_grad
            total_grad = adv_temp - image_tensor
            total_grad = torch.clamp(total_grad, -epsilon, epsilon)
            x_adv = image_tensor + total_grad
            img_variable.data = x_adv
        output_adv = model.forward(img_variable)
        x_adv_pred = labels[torch.max(output_adv.data, 1)[1][0].item()]
        output_adv_probs = F.softmax(output_adv, dim=1)
        x_adv_pred_prob = round((torch.max(output_adv_probs.data, 1)[0][0].item()) * 100, 4)
        imagetensor.append(image_tensor)
        variabledata.append(img_variable.data)
        grads.append(total_grad)
        epsilons.append(epsilon)
        xpreds.append(x_pred)
        xadvpred.append(x_adv_pred)
        xpredprob.append(x_pred_prob)
        xadvpredprob.append(x_adv_pred_prob)
    return imagetensor, variabledata, grads, epsilons, xpreds, xadvpred, xpredprob, xadvpredprob, outnames


def get_gray_visualize(x, classes, probs, outputs, adv):
    """
    Visualize the Adversarial Example and Prediction. Mis-classification label set as Space shuttle for all images
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(len(adv)):
        adv[i] = transform_input(adv[i])
        adv[i] = adv[i].squeeze(0)
        adv[i] = adv[i].mul(torch.FloatTensor(std).view(3, 1, 1)).add(
            torch.FloatTensor(mean).view(3, 1, 1)).numpy()
        adv[i] = np.transpose(adv[i], (1, 2, 0))
        adv[i] = np.clip(adv[i], 0, 1)

    for i in range(len(x)):
        x[i] = x[i].squeeze(0)
        x[i] = x[i].mul(torch.FloatTensor(std).view(3, 1, 1)).add(
            torch.FloatTensor(mean).view(3, 1, 1)).numpy()
        x[i] = np.transpose(x[i], (1, 2, 0))  # C X H X W  ==>   H X W X C
        x[i] = np.clip(x[i], 0, 1)
        plt.imsave(outputs[i], x[i])

    figure, ax = plt.subplots(4, 2, figsize=(18, 8))
    ax[0, 0].imshow(adv[0])
    ax[0, 0].set_title('Adversarial', fontsize=10)

    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    ax[2, 0].axis('off')
    ax[2, 1].axis('off')
    ax[3, 0].axis('off')
    ax[3, 1].axis('off')
    ax[0, 1].imshow(x[0])
    ax[0, 1].set_title('Transformed', fontsize=10)

    ax[0, 0].text(0.5, -0.13, "Prediction: Space Shuttle ", size=6,
                  ha="center",
                  transform=ax[0, 0].transAxes)

    ax[0, 1].text(0.5, -0.13, "Prediction: {} Probability: {}".format(classes[0], probs[0]), size=6, ha="center",
                  transform=ax[0, 1].transAxes)

    ax[1, 0].imshow(adv[1])

    ax[1, 1].imshow(x[1])

    ax[1, 0].text(0.5, -0.13, "Prediction: Space Shuttle ", size=6,
                  ha="center",
                  transform=ax[1, 0].transAxes)

    ax[1, 1].text(0.5, -0.13, "Prediction: {} Probability: {}".format(classes[1], probs[1]), size=6, ha="center",
                  transform=ax[1, 1].transAxes)

    ax[2, 0].imshow(adv[2])

    ax[2, 1].imshow(x[2])

    ax[2, 0].text(0.5, -0.13, "Prediction: Space Shuttle ", size=6,
                  ha="center",
                  transform=ax[2, 0].transAxes)

    ax[2, 1].text(0.5, -0.13, "Prediction: {} Probability: {}".format(classes[2], probs[2]), size=6, ha="center",
                  transform=ax[2, 1].transAxes)

    ax[3, 0].imshow(adv[3])

    ax[3, 1].imshow(x[3])

    ax[3, 0].text(0.5, -0.13, "Prediction: Space Shuttle ", size=6,
                  ha="center",
                  transform=ax[3, 0].transAxes)

    ax[3, 1].text(0.5, -0.13, "Prediction: {} Probability: {}".format(classes[3], probs[3]), size=6, ha="center",
                  transform=ax[3, 1].transAxes)

    plt.show()


def visualize(x, x_adv, x_grad, epsilon, clean_pred, adv_pred, clean_prob, adv_prob, outputs):
    """
    Visualize generation of the Adversarial Example and Prediction.
    """
    for i in range(len(x)):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[i] = x[i].squeeze(0)
        x[i] = x[i].mul(torch.FloatTensor(std).view(3, 1, 1)).add(
            torch.FloatTensor(mean).view(3, 1, 1)).numpy()
        x[i] = np.transpose(x[i], (1, 2, 0))
        x[i] = np.clip(x[i], 0, 1)

        x_adv[i] = x_adv[i].squeeze(0)
        x_adv[i] = x_adv[i].mul(torch.FloatTensor(std).view(3, 1, 1)).add(
            torch.FloatTensor(mean).view(3, 1, 1)).numpy()
        x_adv[i] = np.transpose(x_adv[i], (1, 2, 0))
        x_adv[i] = np.clip(x_adv[i], 0, 1)

        plt.imsave(outputs[i], x_adv[i])

        x_grad[i] = x_grad[i].squeeze(0).numpy()
        x_grad[i] = np.transpose(x_grad[i], (1, 2, 0))
        x_grad[i] = np.clip(x_grad[i], 0, 1)
        x_grad[i] = ((x_grad[i] - x_grad[i].min()) / (x_grad[i].max() - x_grad[i].min())) * 255

    figure, ax = plt.subplots(4, 3, figsize=(18, 8))
    ax[0, 0].imshow(x[0])
    ax[0, 0].set_title('Clean Example', fontsize=15)

    ax[0, 1].imshow(x_grad[0])
    ax[0, 1].set_title('Perturbation', fontsize=15)
    ax[0, 1].set_yticklabels([])
    ax[0, 1].set_xticklabels([])
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])

    ax[0, 2].imshow(x_adv[0])

    ax[0, 2].set_title('Adversarial Example', fontsize=15)

    ax[0, 0].axis('off')
    ax[0, 2].axis('off')

    ax[0, 0].text(0.5, -0.13, "Prediction: {} Probability: {}".format(clean_pred[0], clean_prob[0]), size=10,
                  ha="center",
                  transform=ax[0, 0].transAxes)

    ax[0, 2].text(0.5, -0.13, "Prediction: {} Probability: {}".format(adv_pred[0], adv_prob[0]), size=10, ha="center",
                  transform=ax[0, 2].transAxes)

    ax[1, 0].imshow(x[1])

    ax[1, 1].imshow(x_grad[1])
    ax[1, 1].set_yticklabels([])
    ax[1, 1].set_xticklabels([])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])

    ax[1, 2].imshow(x_adv[1])

    ax[1, 0].axis('off')
    ax[1, 2].axis('off')

    ax[1, 0].text(0.5, -0.13, "Prediction: {} Probability: {}".format(clean_pred[1], clean_prob[1]), size=10,
                  ha="center",
                  transform=ax[1, 0].transAxes)

    ax[1, 2].text(0.5, -0.13, "Prediction: {} Probability: {}".format(adv_pred[1], adv_prob[1]), size=10, ha="center",
                  transform=ax[1, 2].transAxes)

    ax[2, 0].imshow(x[2])

    ax[2, 1].imshow(x_grad[2])
    ax[2, 1].set_yticklabels([])
    ax[2, 1].set_xticklabels([])
    ax[2, 1].set_xticks([])
    ax[2, 1].set_yticks([])

    ax[2, 2].imshow(x_adv[2])

    ax[2, 0].axis('off')
    ax[2, 2].axis('off')

    ax[2, 0].text(0.5, -0.13, "Prediction: {} Probability: {}".format(clean_pred[2], clean_prob[2]), size=10,
                  ha="center",
                  transform=ax[2, 0].transAxes)

    ax[2, 2].text(0.5, -0.13, "Prediction: {} Probability: {}".format(adv_pred[2], adv_prob[2]), size=10, ha="center",
                  transform=ax[2, 2].transAxes)

    ax[3, 0].imshow(x[3])

    ax[3, 1].imshow(x_grad[3])

    ax[3, 1].set_yticklabels([])
    ax[3, 1].set_xticklabels([])
    ax[3, 1].set_xticks([])
    ax[3, 1].set_yticks([])

    ax[3, 2].imshow(x_adv[3])

    ax[3, 0].axis('off')
    ax[3, 2].axis('off')

    ax[3, 0].text(0.5, -0.13, "Prediction: {} Probability: {}".format(clean_pred[3], clean_prob[3]), size=10,
                  ha="center",
                  transform=ax[3, 0].transAxes)

    ax[3, 2].text(0.5, -0.13, "Prediction: {} Probability: {}".format(adv_pred[3], adv_prob[3]), size=10, ha="center",
                  transform=ax[3, 2].transAxes)

    plt.show()


def test_model(images, adversarials):
    """
    Test the efficieny of the black box attack on Resnet50
    """
    model = models.resnet50(pretrained=True)
    model.eval()
    outputs = ['out1.jpg', 'out2.jpg', 'out3.jpg', 'out4.jpg']

    og_images, changed_images, classes, prob = [], [], [], []
    with torch.no_grad():
        for image in images:
            transformed_image = transform_adv(image)
            changed_images.append(transformed_image)
            transformed_image = transformed_image
            output = model(transformed_image)
            _, index = torch.sort(output, descending=True)
            percentage = torch.nn.functional.softmax(output, dim=1)[0]
            label_idx = index[0][:1].item()
            labels_link = "https://savan77.github.io/blog/labels.json"
            labels_json = requests.get(labels_link).json()
            labels = {int(idx): label for idx, label in labels_json.items()}
            x_pred = labels[label_idx]
            output_probs = F.softmax(output, dim=1)
            x_pred_prob = round((torch.max(output_probs.data, 1)[0][0].item()) * 100, 4)
            print(x_pred, "prediction", x_pred_prob, "probability")
            classes.append(x_pred)
            prob.append(x_pred_prob)
    get_gray_visualize(changed_images, classes, prob, outputs, adversarials)


def main():
    """
    Driver Function
    """
    model = models.vgg16(pretrained=True)
    input_image1 = r'D:\peppers.jpg'
    input_image2 = r'D:\Abyssinian.jpg'
    input_image3 = r'D:\beagle.jpg'
    input_image4 = r'D:\Siamese.jpg'
    images = [input_image1, input_image2, input_image3, input_image4]
    out_image1 = "Adv_1.jpg"
    out_image2 = "Adv_2.jpg"
    out_image3 = "Adv_3.jpg"
    out_image4 = "Adv_4.jpg"
    out_images = [out_image1, out_image2, out_image3, out_image4]
    targets = [812, 812, 812, 812]
    one, two, three, four, five, six, seven, eight, nine = gen_adv(images, targets, model, out_images)
    visualize(one, two, three, four, five, six, seven, eight, nine)
    test_model(out_images, out_images)


if __name__ == '__main__':
    main()
