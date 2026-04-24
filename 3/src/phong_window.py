import os.path
import csv
import random

import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44

from base_window import BaseWindow


class PhongWindow(BaseWindow):

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.frame = 0
        self.max_frames = 3000
        self.dataset_info = []

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        self.model_matrix = self.program["model_matrix"]
        self.material_diffuse = self.program["material_diffuse"]
        self.material_shininess = self.program["material_shininess"]
        self.light_position = self.program["light_position"]
        self.camera_position = self.program["camera_position"]

    def on_render(self, time: float, frame_time: float):
        if self.frame >= self.max_frames:
            if self.output_path and len(self.dataset_info) > 0:
                csv_path = os.path.join(self.output_path, 'dataset.csv')
                keys = self.dataset_info[0].keys() 
                
                with open(csv_path, 'w', newline='') as output_file:
                    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(self.dataset_info)
                    
                print(f"Wygenerowano {self.max_frames} obrazów. Zapisano metadane do {csv_path}")
                self.dataset_info = [] 
            
            self.wnd.close()
            return
        
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        camera_position = [5.0, 5.0, 15.0]

        valid_position = False
        while not valid_position:
            model_translation = [
                random.uniform(-20.0, 20.0),
                random.uniform(-20.0, 20.0),
                random.uniform(-20.0, 20.0)
            ]
            dist_to_camera = np.linalg.norm(np.array(model_translation) - np.array(camera_position))
            if dist_to_camera > 2.5:
                valid_position = True

        color_r = random.uniform(0.0, 255.0)
        color_g = random.uniform(0.0, 255.0)
        color_b = random.uniform(0.0, 255.0)
        material_diffuse = [color_r / 255.0, color_g / 255.0, color_b / 255.0]

        material_shininess = random.uniform(3.0, 20.0)

        light_position = [
            random.uniform(-20.0, 20.0),
            random.uniform(-20.0, 20.0),
            random.uniform(-20.0, 20.0)
        ]
        
        model_matrix = Matrix44.from_translation(model_translation)
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            camera_position,
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )

        model_view_projection = proj * lookat * model_matrix

        self.model_view_projection.write(model_view_projection.astype('f4').tobytes())
        self.model_matrix.write(model_matrix.astype('f4').tobytes())
        self.material_diffuse.write(np.array(material_diffuse, dtype='f4').tobytes())
        self.material_shininess.write(np.array([material_shininess], dtype='f4').tobytes())
        self.light_position.write(np.array(light_position, dtype='f4').tobytes())
        self.camera_position.write(np.array(camera_position, dtype='f4').tobytes())

        self.vao.render()
        if self.output_path:
            img = (
                Image.frombuffer('RGBA', self.wnd.size, self.wnd.fbo.read(components=4))
                     .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            )
            img_name = f'image_{self.frame:04}.png'
            img.save(os.path.join(self.output_path, img_name))
            
            self.dataset_info.append({
                "image_file": img_name,
                "model_tx": model_translation[0],
                "model_ty": model_translation[1],
                "model_tz": model_translation[2],
                "diffuse_r": color_r,
                "diffuse_g": color_g,
                "diffuse_b": color_b,
                "shininess": material_shininess,
                "light_px": light_position[0],
                "light_py": light_position[1],
                "light_pz": light_position[2]
            })
            
            self.frame += 1
            if self.frame % 100 == 0:
                print(f"Wygenerowano {self.frame}/{self.max_frames}...")
