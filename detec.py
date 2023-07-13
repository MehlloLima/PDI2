import numpy as np
import imutils
import cv2

# Cria dicionario com os limites de cor no espaço de cor HSV
lower = {'vermelho': (166, 84, 141),
         'azul': (97, 100, 117),
         'amarelo': (23, 59, 119)}

upper = {'vermelho': (186, 255, 255),
         'azul': (117, 255, 255),
         'amarelo': (54, 255, 255)}

# Dicionario com as cores dos circulos
colors = {'vermelho': (0, 0, 255),
          'azul': (255, 0, 0),
          'amarelo': (0, 255, 217)}

# Define a cor a ser detectada
key = "amarelo"


def decte(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Constroi as mascaras para a cor indicada
    # E realiza dilatações e erosoes para evitar areas muito pequenas
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.inRange(hsv, lower[key], upper[key])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Acha os contornos na mascara e o centro para o circulo
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # um laço para executar em cada objeto detectado
    for c in cnts:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Chega se o tamanho do raio é maior que 1
        if radius > 1:
            # desenha os circulos na imagem ao redor do objeto detectado
            cv2.circle(img, (int(x), int(y)),
                    int(radius), colors[key], 2)
            cv2.putText(img, "Objeto " + key, (int(x-radius), int(y - radius)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[key], 2)
    return img
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    
    frame = decte(frame)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
