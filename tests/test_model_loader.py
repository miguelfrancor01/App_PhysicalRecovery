from unittest.mock import patch, MagicMock

from pose_module.model_loader import load_models


@patch("pose_module.model_loader.VitPoseForPoseEstimation")
@patch("pose_module.model_loader.RTDetrForObjectDetection")
@patch("pose_module.model_loader.AutoProcessor")
def test_load_models_carga_modelos_y_procesadores(mock_processor, mock_det_model, mock_pose_model):
    """
    Verifica que load_models:
    - cargue los processors
    - cargue los modelos
    - envíe los modelos al device correcto
    - retorne los 4 objetos esperados
    """

    # Simular processors
    det_processor = MagicMock()
    pose_processor = MagicMock()

    mock_processor.from_pretrained.side_effect = [det_processor, pose_processor]

    # Simular modelos
    det_model = MagicMock()
    pose_model = MagicMock()

    mock_det_model.from_pretrained.return_value = det_model
    mock_pose_model.from_pretrained.return_value = pose_model

    det_model.to.return_value = det_model
    pose_model.to.return_value = pose_model

    device = "cpu"

    result = load_models(device)

    # Verificar llamadas
    assert mock_processor.from_pretrained.call_count == 2
    mock_det_model.from_pretrained.assert_called_once()
    mock_pose_model.from_pretrained.assert_called_once()

    det_model.to.assert_called_once_with(device)
    pose_model.to.assert_called_once_with(device)

    # Verificar retorno
    assert result == (det_processor, det_model, pose_processor, pose_model)