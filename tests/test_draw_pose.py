import numpy as np

from pose_module.draw_pose import draw_pose


def test_draw_pose_dibuja_keypoints_y_conexiones_respetando_threshold():
    """
    Verifica que draw_pose:
    1. conserve el tamaño de la imagen,
    2. modifique la imagen al dibujar keypoints/conexiones válidas,
    3. no dibuje un keypoint cuyo score esté por debajo del threshold.
    """

    # Imagen negra base
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    original = image.copy()

    # 6 keypoints:
    # - 0 a 4 pertenecen a "head"
    # - 5 entra en "arms" y también participa en conexiones válidas de torso/brazos
    keypoints = np.array([
        [10, 10],   # 0
        [20, 10],   # 1
        [10, 20],   # 2
        [20, 20],   # 3
        [15, 30],   # 4
        [60, 60],   # 5  -> este lo dejamos bajo threshold
    ], dtype=np.float32)

    scores = np.array([
        0.95,  # 0
        0.95,  # 1
        0.95,  # 2
        0.95,  # 3
        0.95,  # 4
        0.10,  # 5 -> no debería dibujarse
    ], dtype=np.float32)

    result = draw_pose(image, keypoints, scores, threshold=0.3)

    # 1) La imagen resultante debe conservar la misma forma
    assert result.shape == original.shape

    # 2) Debe haber cambios porque sí se dibujan varios keypoints/conexiones válidas
    assert not np.array_equal(result, original)

    # 3) El keypoint 5 tiene score bajo, así que en su centro no debería haberse dibujado nada
    x5, y5 = int(keypoints[5][0]), int(keypoints[5][1])
    assert np.array_equal(result[y5, x5], np.array([0, 0, 0], dtype=np.uint8))

    # 4) En cambio, un keypoint válido sí debería haber alterado su zona
    x0, y0 = int(keypoints[0][0]), int(keypoints[0][1])
    assert not np.array_equal(result[y0, x0], np.array([0, 0, 0], dtype=np.uint8))