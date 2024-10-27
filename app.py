import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

def add_gaussian_noise(image, mean=0, std=25):
    gaussian_noise = np.random.normal(mean, std, image.shape).astype('uint8')
    noisy_image = np.clip(image + gaussian_noise, 0, 255)
    return noisy_image

def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def rotate_image(image, angle):
    return image.rotate(angle)

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_hue(image, hue_factor):
    hsv_image = image.convert('HSV')
    h, s, v = hsv_image.split()
    
    np_h = np.array(h)
    np_h = (np_h + hue_factor) % 256
    h = Image.fromarray(np_h.astype('uint8'))

    hsv_image = Image.merge('HSV', (h, s, v))
    return hsv_image.convert('RGB')

def apply_gaussian_blur(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius))

def convert_to_grayscale(image):
    return image.convert('L').convert('RGB')

def apply_perspective_transformation(image):
    width, height = image.size
    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    dst_points = np.float32([[0, 0], [width * 0.9, height * 0.1], [width * 0.1, height], [width, height * 0.9]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_image = cv2.warpPerspective(np.array(image), matrix, (width, height))
    return Image.fromarray(transformed_image.astype(np.uint8))

def add_salt_and_pepper_noise(image, salt_prob=0.15, pepper_prob=0.15):
    noisy_image = np.array(image)
    total_pixels = noisy_image.size
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)
    coords = [np.random.randint(0, i - 1, num_salt) for i in noisy_image.shape]
    noisy_image[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy_image.shape]
    noisy_image[coords[0], coords[1]] = 0
    return Image.fromarray(noisy_image.astype(np.uint8))

def augment_and_save_images(image_path):
    image = Image.open(image_path)
    
    noisy_image = add_gaussian_noise(np.array(image))
    flipped_image = flip_image(image)
    rotated_image = rotate_image(image, 45)
    bright_image = adjust_brightness(image, 1.5)
    hue_image = adjust_hue(image, 50)
    blurred_image = apply_gaussian_blur(image, radius=10)
    grayscale_image = convert_to_grayscale(image)
    perspective_transformation_image = apply_perspective_transformation(image)
    salt_and_pepper_image = add_salt_and_pepper_noise(image)

    # Save augmented images
    Image.fromarray(noisy_image).save(os.path.join(app.config['STATIC_FOLDER'], 'augmented_noisy_image.jpg'))
    flipped_image.save(os.path.join(app.config['STATIC_FOLDER'], 'augmented_flipped_image.jpg'))
    rotated_image.save(os.path.join(app.config['STATIC_FOLDER'], 'augmented_rotated_image.jpg'))
    bright_image.save(os.path.join(app.config['STATIC_FOLDER'], 'augmented_bright_image.jpg'))
    hue_image.save(os.path.join(app.config['STATIC_FOLDER'], 'augmented_hue_image.jpg'))
    blurred_image.save(os.path.join(app.config['STATIC_FOLDER'], 'augmented_blurred_image.jpg'))
    grayscale_image.save(os.path.join(app.config['STATIC_FOLDER'], 'augmented_grayscale_image.jpg'))
    perspective_transformation_image.save(os.path.join(app.config['STATIC_FOLDER'], 'augmented_per_transform_image.jpg'))
    salt_and_pepper_image.save(os.path.join(app.config['STATIC_FOLDER'], 'salt_and_pepper_image.jpg'))

@app.route('/', methods=['GET', 'POST'])
def upload_and_show_results():
    images = []
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)
            augment_and_save_images(file_path)
            images = [
                'augmented_noisy_image.jpg',
                'augmented_flipped_image.jpg',
                'augmented_rotated_image.jpg',
                'augmented_bright_image.jpg',
                'augmented_hue_image.jpg',
                'augmented_blurred_image.jpg',
                'augmented_grayscale_image.jpg',
                'augmented_per_transform_image.jpg',
                'salt_and_pepper_image.jpg'
            ]

    return render_template('index.html', images=images)

if __name__ == '__main__':
    app.run(debug=True)
