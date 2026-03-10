from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import datetime
import os


# =================================
# PALETA DE COLORES MÉDICOS
# =================================
AZUL_PRIMARIO    = colors.HexColor("#1A3C6E")   # Azul oscuro institucional
AZUL_SECUNDARIO  = colors.HexColor("#2E6DA4")   # Azul medio
AZUL_CLARO       = colors.HexColor("#D6E8F7")   # Azul muy claro para fondos
GRIS_TEXTO       = colors.HexColor("#2C2C2C")   # Casi negro para texto
GRIS_SUAVE       = colors.HexColor("#F4F7FB")   # Fondo de secciones
BLANCO           = colors.white
VERDE_EXITO      = colors.HexColor("#1E7E4A")
ACENTO           = colors.HexColor("#4A90D9")   # Azul brillante para detalles


def _draw_header(c, width, height):
    """Dibuja el encabezado institucional en la parte superior."""

    # Banda azul superior
    c.setFillColor(AZUL_PRIMARIO)
    c.rect(0, height - 75, width, 75, fill=True, stroke=False)

    # Línea decorativa
    c.setFillColor(ACENTO)
    c.rect(0, height - 78, width, 3, fill=True, stroke=False)

    # Título

    title = "App Physical Recovery"

    c.setFillColor(BLANCO)
    c.setFont("Helvetica-Bold", 18)

    title_x = 40
    title_y = height - 38

    c.drawString(title_x, title_y, title)

    # =========================
    # Logo a la derecha del título
    # =========================
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    logo_path = os.path.join(BASE_DIR, "logos", "Pdf logo.png")

    if os.path.exists(logo_path):

        logo_size = 40

        text_width = c.stringWidth(title, "Helvetica-Bold", 18)

        logo_x = title_x + text_width + 12

        # centra el logo con el texto
        logo_y = title_y - (logo_size / 2) 

        c.drawImage(
            logo_path,
            logo_x,
            logo_y,
            width=logo_size,
            height=logo_size,
            mask='auto'
        )

    # =========================
    # Subtítulo
    # =========================
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#A8C8E8"))
    c.drawString(40, height - 56, "Sistema de Análisis Biomecánico con IA")

    # =========================
    # Fecha y nota de generación automática a la derecha
    # =========================
    c.setFont("Helvetica", 9)
    c.setFillColor(BLANCO)

    fecha = datetime.datetime.now().strftime("%d/%m/%Y  %H:%M")
    c.drawRightString(width - 40, height - 38, fecha)

    c.setFillColor(colors.HexColor("#A8C8E8"))
    c.drawRightString(width - 40, height - 55, "Reporte generado automáticamente")

def _draw_footer(c, width):
    """Dibuja el pie de página."""
    c.setFillColor(AZUL_PRIMARIO)
    c.rect(0, 0, width, 30, fill=True, stroke=False)
    c.setFillColor(colors.HexColor("#A8C8E8"))
    c.setFont("Helvetica", 8)
    c.drawCentredString(width / 2, 11, "Documento generado por App Physical Recovery  •  Uso clínico interno")


def _section_title(c, x, y, text, width):
    """Dibuja un título de sección con banda azul."""
    c.setFillColor(AZUL_CLARO)
    c.rect(x - 8, y - 6, width - x - 32, 22, fill=True, stroke=False)

    # Barra lateral izquierda
    c.setFillColor(AZUL_SECUNDARIO)
    c.rect(x - 8, y - 6, 4, 22, fill=True, stroke=False)

    c.setFillColor(AZUL_PRIMARIO)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x + 4, y + 4, text.upper())

    return y - 30


def _info_row(c, x, y, label, value, col_width=220):
    """Dibuja una fila de información etiqueta: valor."""
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(AZUL_SECUNDARIO)
    c.drawString(x, y, label)

    c.setFont("Helvetica", 10)
    c.setFillColor(GRIS_TEXTO)
    c.drawString(x + col_width, y, value)

    # Línea divisoria suave
    c.setStrokeColor(colors.HexColor("#D0DCE8"))
    c.setLineWidth(0.4)
    c.line(x, y - 5, x + col_width + 180, y - 5)

    return y - 20


def generate_report(session_data, exercise_results, image_path=None):

    data = session_data.get_summary()

    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/exercise_report_{timestamp}.pdf"

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    MARGIN_LEFT  = 40
    MARGIN_RIGHT = width - 40
    CONTENT_W    = MARGIN_RIGHT - MARGIN_LEFT

    # ── ENCABEZADO ──────────────────────────────────────────────
    _draw_header(c, width, height)
    _draw_footer(c, width)

    y = height - 95

    # ── TÍTULO DEL REPORTE ──────────────────────────────────────
    c.setFillColor(AZUL_PRIMARIO)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(MARGIN_LEFT, y, "Reporte de Evaluación de Ejercicio")

    c.setFillColor(AZUL_SECUNDARIO)
    c.setFont("Helvetica", 10)
    c.drawString(MARGIN_LEFT, y - 16, "Análisis de movimiento asistido por visión computacional")

    # Línea separadora
    c.setStrokeColor(ACENTO)
    c.setLineWidth(1.5)
    c.line(MARGIN_LEFT, y - 24, MARGIN_RIGHT, y - 24)

    y -= 44

    # ── SECCIÓN: INFORMACIÓN DEL PACIENTE ───────────────────────
    y = _section_title(c, MARGIN_LEFT, y, "Información del Paciente", width)

    y = _info_row(c, MARGIN_LEFT, y, "Nombre:",     "Adrian Felipe Vargas Rojas")
    y = _info_row(c, MARGIN_LEFT, y, "Edad:",       "26 años")
    y = _info_row(c, MARGIN_LEFT, y, "Sesión N°:",  "3")

    # Padecimiento con wrapping manual
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(AZUL_SECUNDARIO)
    c.drawString(MARGIN_LEFT, y, "Padecimiento:")
    c.setFont("Helvetica", 10)
    c.setFillColor(GRIS_TEXTO)

    padecimiento = "Tendinopatía del manguito rotador con limitación funcional en elevación del brazo izquierdo."
    # Partimos el texto en dos líneas
    linea1 = padecimiento[:55]
    linea2 = padecimiento[55:]
    c.drawString(MARGIN_LEFT + 220, y, linea1)
    if linea2:
        c.drawString(MARGIN_LEFT + 220, y - 14, linea2)
        y -= 14

    c.setStrokeColor(colors.HexColor("#D0DCE8"))
    c.setLineWidth(0.4)
    c.line(MARGIN_LEFT, y - 5, MARGIN_LEFT + 400, y - 5)
    y -= 28

    # ── SECCIÓN: RESULTADOS DEL EJERCICIO ───────────────────────
    y = _section_title(c, MARGIN_LEFT, y, "Resultados del Ejercicio", width)

    y = _info_row(c, MARGIN_LEFT, y, "Tipo de ejercicio:",     data.get("exercise_name", "Elevaciones Frontales con bastón"))
    y = _info_row(c, MARGIN_LEFT, y, "Duración de la sesión:", f"{data.get('duration_seconds', 0)} segundos")
    y = _info_row(c, MARGIN_LEFT, y, "FPS promedio:",          f"{data.get('avg_fps', 0)} fps")
    y = _info_row(c, MARGIN_LEFT, y, "Repeticiones detectadas:", str(exercise_results['repetitions_detected']))

    y -= 10

    # ── TABLA: ÁNGULOS Y PUNTUACIONES ───────────────────────────
    y = _section_title(c, MARGIN_LEFT, y, "Detalle por Repetición", width)

    # Encabezados de tabla
    col1_x = MARGIN_LEFT
    col2_x = MARGIN_LEFT + 200
    col3_x = MARGIN_LEFT + 370

    # Fondo encabezado tabla
    c.setFillColor(AZUL_SECUNDARIO)
    c.rect(col1_x - 4, y - 2, CONTENT_W + 8, 18, fill=True, stroke=False)
    c.setFillColor(BLANCO)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(col1_x + 4,  y + 3, "Repetición")
    c.drawString(col2_x + 4,  y + 3, "Ángulo alcanzado")
    c.drawString(col3_x + 4,  y + 3, "Puntuación")
    y -= 20

    angles = exercise_results.get("angles", [])
    scores = exercise_results.get("scores", [])
    max_rows = max(len(angles), len(scores))

    for i in range(max_rows):
        # Fondo alternado
        if i % 2 == 0:
            c.setFillColor(GRIS_SUAVE)
            c.rect(col1_x - 4, y - 4, CONTENT_W + 8, 18, fill=True, stroke=False)

        c.setFont("Helvetica", 10)
        c.setFillColor(GRIS_TEXTO)
        c.drawString(col1_x + 4, y + 2, f"Rep. {i + 1}")

        if i < len(angles):
            c.drawString(col2_x + 4, y + 2, f"{angles[i]:.2f}°")
        else:
            c.drawString(col2_x + 4, y + 2, "—")

        if i < len(scores):
            score_val = scores[i]
            # Color según puntaje
            if score_val >= 80:
                c.setFillColor(VERDE_EXITO)
            elif score_val >= 50:
                c.setFillColor(AZUL_SECUNDARIO)
            else:
                c.setFillColor(colors.HexColor("#B94040"))
            c.drawString(col3_x + 4, y + 2, f"{score_val:.2f}%")
        else:
            c.drawString(col3_x + 4, y + 2, "—")

        y -= 18

    y -= 10

    # ── PUNTUACIÓN FINAL ────────────────────────────────────────
    score_final = exercise_results['final_score']

    c.setFillColor(AZUL_PRIMARIO)
    c.rect(MARGIN_LEFT - 8, y - 8, CONTENT_W + 16, 30, fill=True, stroke=False)
    c.setFillColor(BLANCO)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(MARGIN_LEFT + 4, y + 6, f"Puntuación final del ejercicio:")

    # Color dinámico del puntaje
    if score_final >= 80:
        c.setFillColor(colors.HexColor("#7FFFC0"))
    elif score_final >= 50:
        c.setFillColor(colors.HexColor("#FFE97F"))
    else:
        c.setFillColor(colors.HexColor("#FF9E9E"))

    c.setFont("Helvetica-Bold", 14)
    c.drawRightString(MARGIN_RIGHT - 8, y + 5, f"{score_final:.2f}%")

    y -= 40

    # ── IMAGEN DEL EJERCICIO ────────────────────────────────────
    if image_path and os.path.exists(image_path):
        img_h = 220
        img_w = 380

        # Si no hay espacio suficiente, nueva página
        if y - img_h < 60:
            c.showPage()
            _draw_header(c, width, height)
            _draw_footer(c, width)
            y = height - 95

        y = _section_title(c, MARGIN_LEFT, y, "Captura del Ejercicio Analizado", width)

        img_h = 220
        img_w = 380

        # Sombra/borde de la imagen
        c.setFillColor(colors.HexColor("#C8D8E8"))
        c.rect(MARGIN_LEFT + 3, y - img_h - 3, img_w, img_h, fill=True, stroke=False)

        c.drawImage(image_path, MARGIN_LEFT, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True)

        # Marco azul alrededor de la imagen
        c.setStrokeColor(AZUL_SECUNDARIO)
        c.setLineWidth(1.5)
        c.rect(MARGIN_LEFT, y - img_h, img_w, img_h, fill=False, stroke=True)

        y -= img_h + 20

    c.save()
    print(f"\nReporte guardado en: {filename}\n")