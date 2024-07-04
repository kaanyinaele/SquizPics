import os
from flask import Flask, request, render_template, redirect, url_for, flash
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploaded')
app.config['COMPRESSED_FOLDER'] = os.path.join(os.path.dirname(__file__), 'compressed')
app.secret_key = 'supersecretkey'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to compress image
def compress_image(image_path, quality_factor):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("The image could not be loaded. Please check the file format.")

        h, w = img.shape
        compressed_img = np.zeros((h, w), np.float32)
        block_size = 8

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = img[i:i+block_size, j:j+block_size]
                dct_block = cv2.dct(np.float32(block) / 255.0)
                quantized_block = np.round(dct_block / quality_factor)
                compressed_img[i:i+block_size, j:j+block_size] = quantized_block

        return compressed_img

    except Exception as e:
        app.logger.error(f'Error compressing image: {e}')
        return str(e)

# Function to decompress image
def decompress_image(compressed_img, quality_factor):
    try:
        h, w = compressed_img.shape
        decompressed_img = np.zeros((h, w), np.float32)
        block_size = 8

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                quantized_block = compressed_img[i:i+block_size, j:j+block_size]
                dequantized_block = quantized_block * quality_factor
                idct_block = cv2.idct(dequantized_block)
                decompressed_img[i:i+block_size, j:j+block_size] = idct_block

        decompressed_img = np.clip(decompressed_img * 255, 0, 255)
        decompressed_img = decompressed_img.astype(np.uint8)
        return decompressed_img

    except Exception as e:
        app.logger.error(f'Error decompressing image: {e}')
        return str(e)

def handle_error(e):
    app.logger.error(f'Error: {e}')
    flash(f'Error: {e}')
    return redirect(url_for('upload_file'))

@app.errorhandler(Exception)
def handle_exception(e):
    return handle_error(e)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            uploaded_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            compressed_filepath = os.path.join(app.config['COMPRESSED_FOLDER'], filename)
            try:
                file.save(uploaded_filepath)
                app.logger.info(f'Uploaded file saved: {uploaded_filepath}')
                quality_factor = 10
                compressed_img = compress_image(uploaded_filepath, quality_factor)
                cv2.imwrite(compressed_filepath, compressed_img)
                app.logger.info(f'Compressed file saved: {compressed_filepath}')
                decompressed_img = decompress_image(compressed_img, quality_factor)
                return redirect(url_for('uploaded_file', filename=filename))
            except Exception as e:
                app.logger.error(f'Error processing image: {e}')
                flash(f'Error processing image: {e}')
                return redirect(request.url)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    uploads_dir = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    compressed_dir = os.path.join(app.root_path, app.config['COMPRESSED_FOLDER'])
    if os.path.exists(os.path.join(uploads_dir, filename)):
        return redirect(url_for('static', filename=os.path.join('uploads', filename)), code=301)
    elif os.path.exists(os.path.join(compressed_dir, filename)):
        return redirect(url_for('static', filename=os.path.join('compressed', filename)), code=301)
    else:
        return handle_error(f'File {filename} not found')

if __name__ == '__main__':
    app.run(debug=True)
