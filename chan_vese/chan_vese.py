import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools

def setParams(new_mu, new_nu, new_eta, new_time_step, new_epsilon):
    global mu, nu_param, eta, time_step, epsilon
    try:
        mu = float(new_mu)
    except ValueError:
        pass
    try:
        nu_param = float(new_nu)
    except ValueError:
        pass
    try:
        eta = float(new_eta)
    except ValueError:
        pass
    try:
        time_step = float(new_time_step)
    except ValueError:
        pass
    try:
        epsilon = float(new_epsilon)
    except ValueError:
        pass

def getParams():
    global mu, nu_param, eta, time_step, epsilon
    return mu, nu_param, eta, time_step, epsilon

def setResize(w, h):
    global widht_resize, height_resize
    try:
        w = int(w)
        if w > 0:
            widht_resize = w
    except ValueError:
        pass
    try:
        h = int(h)
        if h > 0:
            height_resize = h
    except ValueError:
        pass

def getResize():
    global widht_resize, height_resize
    return widht_resize, height_resize

def setMaxIterations(iteration):
    global maxIterations
    try:
        maxIterations = int(iteration)
    except ValueError:
        pass

def getMaxIterations():
    global maxIterations
    return maxIterations

def setTolerance(tol):
    global tolerance
    try:
        tolerance = float(tol)
    except ValueError:
        pass

def getTolerance():
    global tolerance
    return tolerance

def setExpPhase(exp_phs):
    global exp_phase
    try:
        if int(exp_phs) > 0:
            exp_phase = int(exp_phs)
    except:
        pass

def getExpPhase():
    global exp_phase
    return exp_phase

def setLambdas(dupla=None):
    global lambdas
    if 'lambdas' not in globals() or lambdas is None or len(lambdas) != 2 ** exp_phase:
        lambdas = np.full(2 ** exp_phase, float(1))
    try:
        if len(dupla) == 2:
            lambdas[int(dupla[0])] = float(dupla[1])
    except:
        pass

def getLambdas():
    global lambdas
    return lambdas

def setReinicialize(restart):
    global reinitialize
    try:
        reinitialize = int(restart)
    except ValueError:
        pass

def getReinicialize():
    global reinitialize
    return reinitialize

def setImagePath(path):
    global img_path
    img_path = path

def init_phi_circle(img, num_phi, total):
    width, height = img.size
    center_x = width // (total + 1) * (num_phi + 1)
    center_y = height // 2
    radius = min(width, height) // 2 - 5
    phi = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            phi[i, j] = radius - math.sqrt((j - center_x) ** 2 + (i - center_y) ** 2)
    return phi

def init_phi_chessboard(img, num_phi, any = None):
    width, height = img.size
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)
    offset = num_phi
    frequency = (np.pi / 5)
    phi = np.sin(frequency * (X - offset)) * np.sin(frequency * (Y - offset))
    return phi

def setInitialFunction(nFunction):
    global init_phi
    if nFunction == 1:
        init_phi = init_phi_circle
    else:
        init_phi = init_phi_chessboard

def initializeParams():
    setResize(100, 100)
    setParams(0.01, 0
              , 0.0001, 0.5, 1)
    setInitialFunction(2)
    setMaxIterations(20)
    setTolerance(0.001)
    setReinicialize(0)
    setExpPhase(1)
    setLambdas()
    setImagePath("../images/gris_espiral.png")

def image_to_L_or_RGB_and_resize(image_path):
    global fig, ax
    img = Image.open(image_path)
    img = img.resize(getResize())

    if img.mode in ["RGBA", "CMYK", "P", "RGB"]:
        img = img.convert('RGB')
    elif img.mode in ["L"]:
        img = img.convert('L')
    else:
        print(f"El modo de imagen '{img.mode}' no es manejado explícitamente ({img.mode} se tratará de convertir a RGB).")
        img = img.convert("RGB")

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img, cmap='gray' if img.mode == 'L' else None)
    ax[0].axis('off')

    return img

def compute_c(img, list_phis):
    img_array = np.array(img)
    num_phis = len(list_phis)
    c = np.empty((2 ** num_phis, 3) if img.mode == "RGB" else (2 ** num_phis))
    c_regions = []

    combinations = list(itertools.product([True, False], repeat=num_phis))
    assigned = np.full(list_phis[0].shape, False, dtype=bool)
    for comb_idx, combination in enumerate(combinations):
        indices_set = set()
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                if assigned[i, j]:
                    continue
                belongs_to_region = True
                for phi_idx, condition in enumerate(combination):
                    phi = list_phis[phi_idx]
                    if (condition and phi[i, j] < 0) or (not condition and phi[i, j] > 0):
                        belongs_to_region = False
                        break
                if belongs_to_region:
                    indices_set.add((i, j))
                    assigned[i, j] = True

        indices_array = np.array(list(indices_set))
        c_regions.append(indices_array)

        if len(indices_array) > 0:
            values_img = img_array[indices_array[:, 0], indices_array[:, 1]]
            if img.mode == "RGB":
                c[comb_idx] = np.mean(values_img, axis=0)
            else:
                c[comb_idx] = np.mean(values_img)
        else:
            c[comb_idx] = np.array([0, 0, 0]) if img.mode == "RGB" else 0
    return c, c_regions

def draw_avarage_image(img, c, c_regions):
    global fig, ax
    img_array = np.array(img)
    for z, c_region in enumerate(c_regions):
        if len(c_region) > 0:
            img_array[c_region[:, 0], c_region[:, 1]] = c[z]
    ax[1].imshow(img_array, cmap='gray', vmin=0, vmax=255 if img.mode == 'L' else None)
    ax[1].axis('off')

def draw_phi_image(img, borders):
    global fig, ax
    img_array = np.array(img)
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 165, 0),
        (128, 0, 128)
    ]
    if img.mode == "L":
        img_array = np.stack((img_array,) * 3, axis=-1)

    if len(borders[0]) > 0:
        for z, between_idx in enumerate(borders):
            img_array[between_idx[:, 0], between_idx[:, 1]] = colors[z]

    ax[2].imshow(img_array)
    ax[2].axis('off')

def updateTitles(ax, iteration, c):
    ax[0].set_title("Original")
    ax[2].set_title(f"Iteración {iteration}")
    grouped_strs = []
    for i in range(0, len(c), 2):
        group = [
            f"c{j + 1} = ({c[j][0]:.0f}, {c[j][1]:.0f}, {c[j][2]:.0f})" if isinstance(c[j], (list, np.ndarray)) and len(
                c[j]) == 3
            else f"c{j + 1} = {c[j]:.0f}"
            for j in range(i, min(i + 2, len(c)))
        ]
        grouped_strs.append(", ".join(group))
    title_str = "\n".join(grouped_strs)
    ax[1].set_title(title_str)

def reinitialization(phi):
    rows, cols = phi.shape
    borders = find_borders(phi)
    dD = np.zeros_like(phi)
    for i in range(rows):
        for j in range(cols):
            if len(borders) >0 and [i,j] in borders:
                continue
            a = phi[i, j] - (phi[i, j - 1] if j > 0 else 0)
            b = (phi[i, j + 1] if j < cols - 1 else 0) - phi[i, j]
            c = phi[i, j] - (phi[i - 1, j] if i > 0 else 0)
            d = (phi[i + 1, j] if i < rows - 1 else 0) - phi[i, j]

            a_p = max(a, 0)
            a_n = min(a, 0)
            b_p = max(b, 0)
            b_n = min(b, 0)
            c_p = max(c, 0)
            c_n = min(c, 0)
            d_p = max(d, 0)
            d_n = min(d, 0)

            if phi[i, j] > 0:
                dD[i, j] = np.sqrt(
                    max(a_p ** 2, b_n ** 2) +
                    max(c_p ** 2, d_n ** 2)) - 1
            elif phi[i, j] < 0:
                dD[i, j] = np.sqrt(
                    max(a_n ** 2, b_p ** 2) +
                    max(c_n ** 2, d_p ** 2)) - 1

    phi = phi - time_step * sussman_sign(phi) * dD
    for [i,j] in borders:
        if len(borders > 0) and [i, j] in borders:
            phi[i,j] = 0
    return phi

def sussman_sign(phi):
    return phi / np.sqrt(phi ** 2 + 1)

def phi_Stationary(phi, phi_past):
    global tolerance, past
    return True if ((np.linalg.norm(phi - phi_past)) / phi.size) < tolerance else False

def euclidean_norm(image_i_j, c):
    return np.linalg.norm(image_i_j - c)

def find_borders(phi):
    borders = []
    for i in range(1, phi.shape[0] - 1):
        for j in range(1, phi.shape[1] - 1):
            if (phi[i, j] == 0 or (phi[i, j] > 0 and (
                    phi[i - 1, j] < 0 or phi[i + 1, j] < 0 or phi[i, j - 1] < 0 or phi[i, j + 1] < 0 or phi[i+1,j+1]<0 or phi[i-1,j-1]<0 or phi[i-1,j+1]<0 or phi[i+1,j-1] <0))): #orphi[i+1,j+1]<0 or phi[i-1,j-1]<0 or phi[i-1,j+1]<0 or phi[i+1,j-1] <0
                borders.append((i, j))
    return np.array(borders)


def chan_vese_segmentation():

    start = time.time()

    img = image_to_L_or_RGB_and_resize(img_path)
    total_layers = 3 if img.mode =="RGB" else 1
    width, height = img.size
    img_array = np.array(img).astype(float)
    exp_phase = getExpPhase()
    list_phis = []
    for i in range(exp_phase):
        phi = init_phi(img,i, exp_phase)
        list_phis.append(phi)
    borders = []
    for i, phi in enumerate(list_phis):
        borders.append(find_borders(phi))

    draw_phi_image(img.copy(), borders)
    c, c_regions = compute_c(img, list_phis)
    draw_avarage_image(img.copy(), c, c_regions)
    updateTitles(ax, 0, c)
    plt.show(block=False)
    plt.pause(1)
    

    for iteration in range(maxIterations):
        count_stationary = 0
        for num_phi, phi in enumerate(list_phis):
            phi_past = phi.copy()
            for i in range(height):
                nl = i - 1
                nr = i + 1
                nl = 0 if nl == -1 else nl
                nr = height - 1 if nr == height else nr
                for j in range(width):
                    nu = j - 1
                    nd = j + 1
                    nu = 0 if nu == -1 else nu
                    nd = width - 1 if nd == width else nd

                    delta = epsilon / (np.pi * (epsilon ** 2 + phi[i, j] ** 2))
                    A_i_j = mu / (
                        np.sqrt(eta ** 2 + (phi[nr, j] - phi[i, j]) ** 2 + ((phi[i, nd] - phi[i, nu]) / 2) ** 2))
                    A_nl_j = mu / (
                        np.sqrt(eta ** 2 + (phi[i, j] - phi[nl, j]) ** 2 + ((phi[nl, nd] - phi[nl, nu]) / 2) ** 2))
                    B_i_j = mu / (
                        np.sqrt(eta ** 2 + ((phi[nr, j] - phi[nl, j]) / 2) ** 2 + (phi[i, j] - phi[nr, j]) ** 2))
                    B_i_nu = mu / (
                        np.sqrt(eta ** 2 + ((phi[nr, nu] - phi[nl, nu]) / 2) ** 2 + (phi[i, nu] - phi[nr, nu]) ** 2))

                    first_term = A_i_j * phi[nr, j] + A_nl_j * phi[nl, j] + B_i_j * phi[i, nd] + B_i_nu * phi[i, nu]
                    denominator = 1 + time_step * delta * (A_i_j + A_nl_j + B_i_j + B_i_nu)

                    intersection = sum(
                        1 for num_phi_z in range(exp_phase) if num_phi_z != num_phi and list_phis[num_phi_z][i, j] > 0)

                    magic = 0
                    combinations = list(itertools.product([True, False], repeat=exp_phase))
                    for comb_idx, combination in enumerate(combinations):
                        sign = 1 if combination[num_phi] else -1
                        sign2 = 1
                        for phi_idx, condition in enumerate(combination):  # Una combinación, por ejemplo (H1,H2,H3^-1)
                            if phi_idx != num_phi:
                                phi_aux = list_phis[phi_idx]
                                if condition:
                                    sign2 *= 1 if phi_aux[i, j] >= 0 else 0
                                else:
                                    sign2 *= 1 - (1 if phi_aux[i, j] >= 0 else 0)
                        magic += -sign * lambdas[comb_idx] * (
                                    (euclidean_norm(img_array[i, j], c[comb_idx])) ** 2) * sign2

                    second_term = -nu_param * (1 - intersection) + magic *1/total_layers

                    phi[i, j] = (phi[i, j] + time_step * delta * (first_term + second_term)) / denominator


            if reinitialize > 0 and ((iteration + 1) % reinitialize == 0):
                for reiniIter in range(0,2):
                    list_phis[num_phi] =reinitialization(phi)
                    phi = list_phis[num_phi]


            if phi_Stationary(phi, phi_past):
                count_stationary += 1

            borders[num_phi] = find_borders(phi)

        c, c_regions = compute_c(img, list_phis)

        if count_stationary == exp_phase:
            draw_phi_image(img.copy(), borders)
            draw_avarage_image(img.copy(), c, c_regions)
            updateTitles(ax, iteration + 1, c)
            end = time.time()
            segmen_time = end - start
            fig.text(0.5, 0.25, "La segmentación a finalizado", ha='center', va='center', fontsize=12)
            fig.text(0.5, 0.20, f"Tiempo transcurrido ", ha='center', va='center', fontsize=12)
            fig.text(0.5, 0.15, f"{segmen_time:.2f} segundos", ha='center', va='center', fontsize=12, color='red')
            plt.show(block=False)
            break

        if iteration > 0 and (((iteration + 1) % 5 == 0) or (iteration + 1 == maxIterations)):
            draw_phi_image(img.copy(), borders)
            draw_avarage_image(img.copy(), c, c_regions)
            updateTitles(ax, iteration + 1, c)
            plt.show(block=False)
            plt.pause(1)

    end = time.time()
    segmen_time = end - start
    fig.text(0.5, 0.25, "La segmentación a finalizado", ha='center', va='center', fontsize=12)
    fig.text(0.5, 0.20, f"Tiempo transcurrido ", ha='center', va='center', fontsize=12)
    fig.text(0.5, 0.15, f"{segmen_time:.2f} segundos", ha='center', va='center', fontsize=12, color='red')

    plt.show()



if __name__ == "__main__":
    initializeParams()
    chan_vese_segmentation()
