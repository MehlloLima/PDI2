import cv2
import numpy as np
import matplotlib.pyplot as plt

# Abre imagem em escala de cinza
img_original = cv2.imread("lena.jpg",0)
img_original = cv2.resize(img_original, (600,600))
h, w = img_original.shape
print(img_original.shape)

##### Convertendo imagem para dominio da Frequencia #####

# Converte a imagem para float 64
img = np.float64(img_original)

# Aplica transformada de Fourier
fft = np.fft.fft2(img)

# Aplica uma mudança de cantos para que a alta frequencia fica ao centro
fft = np.fft.fftshift(fft)

# Abs para evitar negativos
fft_abs = np.abs(fft)

# Usando log para rescalonar entre 0-1
fft_log = 20*np.log10(fft_abs)

# Exibe imagem original
cv2.imshow("Imagem Original", img_original)
cv2.imwrite("Imagem Original.png", img_original)
# Exibe o espectro de Fourier utilizando *255 para voltar a escala 0-255 
cv2.imshow("Espectro de Fourier", np.uint8(255*fft_log/np.max(fft_log)))
cv2.imwrite("Espectro de Fourier.png", np.uint8(255*fft_log/np.max(fft_log)))


##### Filtro ideal #####

# Filtro para frequencia horizontal e vertical
FH = np.arange(-h/2 +1, h/2 +1, 1)
FV = np.arange(-w/2 +1, w/2 +1, 1)

# gera um array/matriz do filtro
[x,y] = np.meshgrid(FH, FV)

# Distancia Maxima do centro do filtro para a borda
D = np.sqrt(x**2 + y**2)
D = D/np.max(D)

# Definir tamanho do circulo interno do filtro, 0,25 do tamanho total
Do = 0.25


##### Gerando o Filtro Passa Baixa #####

# matriz inicial de zeros do tamanho da imagem
Fuv = np.zeros((h,w))
for i in range(h):
    for j in range(w):
        if(D[i,j]<Do):
            Fuv[i,j]=1

# Gerando o Filtro Passa Alta com base no passa baixa
F2uv = 1 - Fuv

# Mostrando os Filtro
cv2.imshow("Filtro Passa Baixa Ideal",Fuv)
cv2.imwrite("Filtro Passa Baixa Ideal.png",Fuv)
cv2.imshow("Filtro Passa Alta Ideal",F2uv)
cv2.imwrite("Filtro Passa Alta Ideal.png",F2uv)

##### Aplicando Filtro Passa Baixa #####

# Aplicando na frequencia
Gxy = Fuv*fft

# Calculo de magnitude
Gxy_abs = np.abs(Gxy)
Gxy_abs = np.uint8(255*Gxy_abs/np.max(Gxy_abs))

cv2.imshow("Espectro da frequencia G Para passa Baixa", Gxy_abs)
cv2.imwrite("Espectro da frequencia G Para passa Baixa.png", Gxy_abs)

##### Aplicar transformada inversa para obter a imagem Filtrada #####

# IFFT2
gxy = np.fft.ifft2(Gxy)
gxy = np.abs(gxy)
gxy = np.uint8(gxy)

#exibe imagem final
cv2.imshow("Imagem Filtrada Passa Baixa", gxy)
cv2.imwrite("Imagem Filtrada Passa Baixa.png", gxy)


##### Aplicando Filtro Passa Alta #####

# Aplicando na frequencia
G2xy = F2uv*fft

# Calculo de magnitude
G2xy_abs = np.abs(G2xy)
G2xy_abs = np.uint8(255*G2xy_abs/np.max(G2xy_abs))

cv2.imshow("Espectro da frequencia G Para passa Alta", G2xy_abs)
cv2.imwrite("Espectro da frequencia G Para passa Alta.png", G2xy_abs)

##### Aplicar transformada inversa para obter a imagem Filtrada #####

# IFFT2
g2xy = np.fft.ifft2(G2xy)
g2xy = np.abs(g2xy)
g2xy = np.uint8(g2xy)

#exibe imagem final
cv2.imshow("Imagem Filtrada Passa Alta", g2xy)
cv2.imwrite("Imagem Filtrada Passa Alta.png", g2xy)


cv2.waitKey(0)