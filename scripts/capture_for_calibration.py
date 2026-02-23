#!/usr/bin/env python3
import cv2
import os
import time
import numpy as np
from datetime import datetime

def capture_images_for_calibration(save_folder, camera_id=0, wait_time=2):
    """
    Captura imágenes para calibración
    
    Args:
        save_folder: Carpeta donde guardar las imágenes
        camera_id: ID de la cámara
        wait_time: Tiempo de espera entre capturas (segundos)
    """
    # Crear carpeta si no existe
    os.makedirs(save_folder, exist_ok=True)
    
    # Inicializar cámara
    cap = cv2.VideoCapture(camera_id)
    
    # Configurar resolución (ajustar según tu cámara)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    print("="*50)
    print("📸 CAPTURADOR DE IMÁGENES PARA CALIBRACIÓN")
    print("="*50)
    print(f"Carpeta de guardado: {save_folder}")
    print("\nControles:")
    print("  - Presiona ESPACIO para capturar imagen")
    print("  - Presiona 'q' para salir")
    print("  - Presiona 'c' para cambiar modo continuo")
    print("="*50)
    
    image_count = len([f for f in os.listdir(save_folder) if f.endswith('.jpg')])
    continuous_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al capturar frame")
            break
        
        # Mostrar información en la imagen
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Imágenes: {image_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Modo: {'CONTINUO' if continuous_mode else 'MANUAL'}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "ESPACIO: capturar | q: salir | c: cambiar modo", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Captura para Calibración', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Espacio - captura manual
            image_count = save_image(frame, save_folder, image_count)
            
        elif key == ord('c'):  # Cambiar modo continuo
            continuous_mode = not continuous_mode
            print(f"🔄 Modo continuo: {'ACTIVADO' if continuous_mode else 'DESACTIVADO'}")
            
        elif key == ord('q'):  # Salir
            break
        
        # Modo continuo
        if continuous_mode:
            time.sleep(wait_time)
            image_count = save_image(frame, save_folder, image_count)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Captura finalizada. Total imágenes: {image_count}")

def save_image(frame, folder, count):
    """Guarda una imagen con nombre secuencial"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"calib_image_{count+1:03d}_{timestamp}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)
    print(f"✅ Imagen guardada: {filename}")
    return count + 1

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Capturador de imágenes para calibración')
    parser.add_argument('--folder', type=str, default='./calibration_images',
                       help='Carpeta para guardar las imágenes')
    parser.add_argument('--camera', type=int, default=0,
                       help='ID de la cámara')
    parser.add_argument('--wait', type=float, default=2.0,
                       help='Tiempo de espera entre capturas en modo continuo')
    
    args = parser.parse_args()
    
    capture_images_for_calibration(
        save_folder=args.folder,
        camera_id=args.camera,
        wait_time=args.wait
    )

if __name__ == '__main__':
    main()