#!/usr/bin/env python3
"""
YOLO pour Reveal.js - Mode Sauvegarde Auto
=========================================

Capture automatique d'images YOLO pour intégration dans Reveal.js
sans serveur de streaming. Les images sont sauvées automatiquement
et peuvent être affichées dans les slides.

Usage: python yolo_reveal_auto.py
"""

import cv2
import torch
from ultralytics import YOLO
import time
import numpy as np
import os
from datetime import datetime

class YOLOAutoCapture:
    def __init__(self, save_interval=0.037, output_dir="yolo_captures"):
        self.save_interval = save_interval  # Intervalle de sauvegarde en secondes (~27 FPS VITESSE MAX)
        self.output_dir = output_dir
        self.setup_output_dir()
        self.setup_yolo()
        self.setup_camera()
        
    def setup_output_dir(self):
        """Créer le dossier de sortie."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print(f"📂 Dossier de sortie: {self.output_dir}")
        
    def setup_yolo(self):
        """Configure YOLO pour performance maximale."""
        print("🚀 Initialisation de YOLO en mode PERFORMANCE MAX...")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📱 Device: {self.device.upper()}")
        
        self.model = YOLO('yolov8n.pt')  # yolov8n = nano = plus rapide
        self.model.to(self.device)
        
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Plus rapide mais moins déterministe
            print("⚡ CUDA optimisé pour vitesse maximale")
        
    def setup_camera(self):
        """Configure la caméra pour performance maximale."""
        print("🔍 Configuration de la caméra pour vitesse maximale...")
        
        for camera_id in range(5):
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.camera = cap
                    # Configuration optimisée pour vitesse maximale
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.camera.set(cv2.CAP_PROP_FPS, 60)  # Demander 60 FPS à la caméra
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer minimal pour réduire la latence
                    print(f"📷 Caméra trouvée: ID {camera_id} - Mode haute vitesse")
                    return
            cap.release()
        
        raise Exception("❌ Aucune caméra trouvée")
    
    def draw_detections(self, frame, results, confidence_threshold=0.5):
        """Dessine les détections."""
        detections_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < confidence_threshold:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    # Couleurs vives
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                    color = colors[class_id % len(colors)]
                    
                    # Rectangle et texte
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    detections_count += 1
        
        return frame, detections_count
    
    def run_auto_capture(self):
        """Mode capture automatique ultra-rapide pour Reveal.js."""
        print("� Démarrage de la capture MAXIMUM SPEED...")
        print(f"⚡ Sauvegarde toutes les {self.save_interval:.3f} secondes (~{1/self.save_interval:.0f} FPS)")
        print("Contrôles: 'q'=quitter, 's'=sauvegarder maintenant")
        
        last_save_time = time.time()
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Détecter les objets avec confidence réduite pour plus de vitesse
                results = self.model(frame, conf=0.3, verbose=False, half=True)  # half=True pour FP16 si supporté
                annotated_frame, detections = self.draw_detections(frame, results, confidence_threshold=0.3)
                
                # Calcul FPS en temps réel
                fps_counter += 1
                current_time = time.time()
                elapsed_fps = current_time - fps_start_time
                if elapsed_fps >= 1.0:
                    actual_fps = fps_counter / elapsed_fps
                    fps_counter = 0
                    fps_start_time = current_time
                else:
                    actual_fps = fps_counter / max(elapsed_fps, 0.001)
                
                # Ajouter timestamp et infos de performance
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Millisecondes
                cv2.putText(annotated_frame, f"YOLO MAX SPEED - {timestamp}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Objets: {detections} | FPS: {actual_fps:.1f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Target: ~{1/self.save_interval:.0f} FPS", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Afficher
                cv2.imshow('YOLO MAX SPEED Capture', annotated_frame)
                
                # Sauvegarde automatique ultra-rapide
                if current_time - last_save_time >= self.save_interval:
                    filename = f"{self.output_dir}/yolo_live.jpg"
                    # Compression JPEG optimisée pour vitesse
                    cv2.imwrite(filename, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    last_save_time = current_time
                
                # Gestion des touches (polling minimal pour ne pas ralentir)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"{self.output_dir}/yolo_{int(time.time())}.jpg"
                    cv2.imwrite(filename, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"📸 Sauvé manuellement: {filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print(f"\n⚠️ Arrêt demandé après {frame_count} frames")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Nettoyage."""
        if hasattr(self, 'camera'):
            self.camera.release()
        cv2.destroyAllWindows()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == "__main__":
    print("=" * 50)
    print("🎯 YOLO Auto-Capture pour Reveal.js")
    print("=" * 50)
    
    try:
        capturer = YOLOAutoCapture(save_interval=0.037)  # ~27 FPS VITESSE MAXIMALE - Optimisé pour RTX 4060
        capturer.run_auto_capture()
    except Exception as e:
        print(f"❌ Erreur: {e}")