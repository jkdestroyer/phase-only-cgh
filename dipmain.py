from pathlib import Path
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np



#
INPUT_PATH = Path("target.jpg")
OUTPUT_DIR = Path("results")

IMAGE_SIZE = (256, 256)
WIDTH, HEIGHT = IMAGE_SIZE

ITERATIONS = [1, 10, 50, 100]
OUTPUT_DIR.mkdir(exist_ok=True)

# FFT 라이브러리를 사용하지 않고, DFT 수식을 직접 구현해보는데 목적을 두어 
# 느리더라도 DFT 행렬 방식으로 구현하였다. 
# 수업시간에 배운 수식을 그대로 코드로 옮겼다. 
def make_fourier_matrix(size, inverse=False):
    """
    DFT 수식 중 exp(-2j*pi*k*n/N) 부분을 행렬로 만드는 함수
    """
    
    n = np.arange(size)
    k = n.reshape(size, 1)

# DFT는 지수에 음수, IDFT는 지수에 양수
    if inverse: 
        sign = 1
    else:
        sign = -1
    
    

    return np.exp(sign * 2j * np.pi * k * n / size)

# 프로그램 시작 후 바로 Fourier matrix 생성

row_dft = make_fourier_matrix(HEIGHT)
col_dft = make_fourier_matrix(WIDTH)

row_idft = make_fourier_matrix(HEIGHT, inverse=True)
col_idft = make_fourier_matrix(WIDTH, inverse=True)



def dft2(field):
    """
    2D Discrete Fourier Transform
    """

    return row_dft @ field @ col_dft
    
def idft2(field):
    """
    2D Inverse Discrete Fourier Transform
    """

    return row_idft @ field @ col_idft/(HEIGHT * WIDTH) #원본의 크기로 되돌리기 위해 나눠줌

# target image  불러오기

target_image = cv2.imread(
    str(INPUT_PATH),
    cv2.IMREAD_GRAYSCALE #이미 흑백 이미지더라도, 컴퓨터가 컬러로 읽어들이는 경우가 있기에 한 번 더 확실하게 하는 용도
)

if target_image is None:
    raise FileNotFoundError(
        "target.jpg가 없습니다."
    )

target_image = cv2.resize( # 입력 이미지를 256*256 로 맞추기 (이것도 교차검증용)
    target_image,
    IMAGE_SIZE,
    interpolation=cv2.INTER_AREA
)


# 픽셀 값을 0~1로 normalize
target_intensity = target_image.astype(np.float64) / 255.0

# 빛의 intensity = amplitude^2
target_amplitude = np.sqrt(target_intensity)



#초기 random phase 생성 (-π ~ π 범위)
rng = np.random.default_rng(0)

initial_phase = rng.uniform(
    -np.pi,
    np.pi,
    IMAGE_SIZE
)

# Phase-only hologram:
# amplitude는 1, phase만 random 값으로 설정
hologram_field = np.exp(1j * initial_phase)


# 결과 저장용
reconstruction_results = {}
phase_results = {}
mse_results = {}


# Gerchberg-Saxton algorithm 
for iteration in range(1, max(ITERATIONS) + 1):

    # Hologram plane → image plane
    image_field = dft2(hologram_field)

    # 현재 image plane의 phase
    image_phase = np.angle(image_field)

    # Target amplitude를 적용하고 phase는 유지
    image_field = (
        target_amplitude
        * np.exp(1j * image_phase)
    )

    # Image plane → hologram plane
    hologram_field = idft2(image_field)

    # Hologram plane에서는 phase만 남김
    hologram_phase = np.angle(hologram_field)

    hologram_field = np.exp(
        1j * hologram_phase
    )

    # 1, 10, 50, 100 iteration에서 결과 저장
    if iteration in ITERATIONS:

        reconstructed_field = dft2(
            hologram_field
        )

        reconstructed_intensity = (
            np.abs(reconstructed_field) ** 2
        )

        # 0~1 범위로 정규화
        reconstructed_intensity = (
            reconstructed_intensity
            / (reconstructed_intensity.max() + 1e-12)
        )

        # Mean Square Error 계산
        mse = np.mean(
            (
                target_intensity
                - reconstructed_intensity
            ) ** 2
        )

        reconstruction_results[iteration] = (
            reconstructed_intensity.copy()
        )

        phase_results[iteration] = (
            hologram_phase.copy()
        )

        mse_results[iteration] = float(mse)

        print(
            f"Iteration {iteration:3d}"
            f" | MSE = {mse:.6f}"
        )


# 최종 phase-only hologram 저장
final_phase = np.angle(hologram_field)

phase_image = (
    (final_phase + np.pi)
    / (2 * np.pi)
    * 255
)

phase_image = np.clip(
    phase_image,
    0,
    255
).astype(np.uint8)

cv2.imwrite(
    str(OUTPUT_DIR / "phase_hologram_final.png"),
    phase_image
)


# reconstruction 결과를 한눈에 비교할 수 있는 최종 이미지 생성

fig, axes = plt.subplots(
    1,
    len(ITERATIONS) + 1,
    figsize=(15, 4)
)

axes[0].imshow(
    target_intensity,
    cmap="gray"
)

axes[0].set_title("Target")
axes[0].axis("off")

for index, iteration in enumerate(
    ITERATIONS  ,
    start=1
):

    axes[index].imshow(
        reconstruction_results[iteration],
        cmap="gray"
    )

    axes[index].set_title(
        f"Iteration {iteration}\n"
        f"MSE={mse_results[iteration]:.5f}"
    )

    axes[index].axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "reconstruction_comparison.png",
    dpi=200
)

plt.close()


# MSE 그래프 생성

iterations = ITERATIONS

mse_values = [
    mse_results[iteration]
    for iteration in ITERATIONS
]

plt.figure(figsize=(6, 4))

plt.plot(
    iterations,
    mse_values,
    marker="o"
)

plt.xlabel("Number of iterations")
plt.ylabel("MSE")
plt.title("MSE according to GS iterations")
plt.grid()

plt.savefig(
    OUTPUT_DIR / "mse_graph.png",
    dpi=200
)

plt.close()


# MSE 결과를 CSV 파일로 저장

with open(
    OUTPUT_DIR / "mse_results.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Iteration",
        "MSE"
    ])

    for iteration in ITERATIONS:

        writer.writerow([
            iteration,
            mse_results[iteration]
        ])


print()
print("실행 완료")
print("results 폴더에서 결과를 확인하세요.")