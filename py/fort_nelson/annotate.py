'''20251203 image annotation tool: positive and negative samples 

saved/updated to shapefile. 

Shapefile attrs include coordinates of selected areas, and original image file name
'''
# ============================================================
# QGIS Annotation Tool – Polygons with Coordinates & Image
# With Color Coding for Positive/Negative Samples
# ============================================================

from qgis.PyQt.QtWidgets import QPushButton, QHBoxLayout, QWidget, QGraphicsSimpleTextItem
from qgis.PyQt.QtGui import QColor
from PyQt5.QtCore import QPointF, QVariant
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsField, QgsGeometry, QgsRectangle,
    QgsWkbTypes, QgsFeature, QgsMapLayer, QgsSymbol, QgsRendererCategory,
    QgsCategorizedSymbolRenderer, QgsFillSymbol
)
from qgis.gui import QgsMapTool, QgsRubberBand
import os

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
OUTPUT_SHP = "/ram/parker/annotation_labels.shp"
DEFAULT_CLASS = "NEGATIVE"

# ------------------------------------------------------------
# REMOVE EXISTING BUTTONS  (FIXED)
# ------------------------------------------------------------
if hasattr(iface, "annotation_button_widget"):
    try:
        parent_tb = iface.annotation_button_widget.parentWidget()
        if parent_tb and hasattr(parent_tb, "removeWidget"):
            parent_tb.removeWidget(iface.annotation_button_widget)
        iface.annotation_button_widget.setParent(None)
        iface.annotation_button_widget.deleteLater()
    except Exception as e:
        print("Could not remove existing annotation widget:", e)
    finally:
        del iface.annotation_button_widget

# ------------------------------------------------------------
# LOAD OR CREATE SHAPEFILE
# ------------------------------------------------------------
if not os.path.exists(OUTPUT_SHP):
    vl = QgsVectorLayer("Polygon?crs=EPSG:4326", "annotations", "memory")
    pr = vl.dataProvider()
    pr.addAttributes([
        QgsField("CLASS", QVariant.String, "string", 10),
        QgsField("SRC_IMAGE", QVariant.String, "string", 200),
        QgsField("COORDS", QVariant.String, "string", 2000),
        QgsField("COORDS_IMG", QVariant.String, "string", 2000)
    ])
    vl.updateFields()
    from qgis.core import QgsVectorFileWriter
    QgsVectorFileWriter.writeAsVectorFormat(
        vl, OUTPUT_SHP, "utf-8", driverName="ESRI Shapefile"
    )

# Load shapefile
annotation_layer = QgsVectorLayer(OUTPUT_SHP, "Annotations", "ogr")
if not annotation_layer.isValid():
    raise Exception("Could not load annotation shapefile!")

# Force CRS to match canvas
annotation_layer.setCrs(iface.mapCanvas().mapSettings().destinationCrs())
QgsProject.instance().addMapLayer(annotation_layer)

# ------------------------------------------------------------
# ENSURE REQUIRED FIELDS
# ------------------------------------------------------------
def ensure_fields(layer):
    provider = layer.dataProvider()
    fields = [f.name() for f in layer.fields()]
    added = False
    for f_name, f_type, f_len in [("CLASS", QVariant.String, 10),
                                  ("SRC_IMAGE", QVariant.String, 200),
                                  ("COORDS", QVariant.String, 2000),
                                  ("COORDS_IMG", QVariant.String, 2000)]:
        if f_name not in fields:
            provider.addAttributes([QgsField(f_name, f_type, "string", f_len)])
            added = True
    if added:
        layer.updateFields()

ensure_fields(annotation_layer)

# ------------------------------------------------------------
# GET TOP RASTER FILENAME AND LAYER
# ------------------------------------------------------------
def get_top_raster_layer():
    """Get the topmost raster layer"""
    layers = QgsProject.instance().layerTreeRoot().layerOrder()
    for lyr in layers:
        if lyr.type() == QgsMapLayer.RasterLayer:
            return lyr
    return None

def get_top_raster_filename():
    lyr = get_top_raster_layer()
    if lyr:
        return os.path.basename(lyr.source())
    return "UNKNOWN"

def geo_to_pixel_coords(raster_layer, geo_points):
    if not raster_layer or not raster_layer.isValid():
        return None
    
    extent = raster_layer.extent()
    width = raster_layer.width()
    height = raster_layer.height()
    
    pixel_coords = []
    for point in geo_points:
        col = int((point.x() - extent.xMinimum()) / extent.width() * width)
        row = int((extent.yMaximum() - point.y()) / extent.height() * height)
        col = max(0, min(col, width - 1))
        row = max(0, min(row, height - 1))
        pixel_coords.append(f"{col},{row}")
    
    return ";".join(pixel_coords)

# ------------------------------------------------------------
# APPLY COLOR-CODED SYMBOLOGY
# ------------------------------------------------------------
def apply_color_symbology(layer):
    positive_symbol = QgsFillSymbol.createSimple({
        'color': '0,255,0,100',
        'outline_color': '0,180,0,255',
        'outline_width': '0.5'
    })
    
    negative_symbol = QgsFillSymbol.createSimple({
        'color': '255,0,0,100',
        'outline_color': '180,0,0,255',
        'outline_width': '0.5'
    })
    
    categories = [
        QgsRendererCategory('POSITIVE', positive_symbol, 'Positive'),
        QgsRendererCategory('NEGATIVE', negative_symbol, 'Negative'),
    ]
    
    renderer = QgsCategorizedSymbolRenderer('CLASS', categories)
    layer.setRenderer(renderer)
    layer.triggerRepaint()

apply_color_symbology(annotation_layer)

# ------------------------------------------------------------
# BACKFILL COORDS_IMG FOR EXISTING FEATURES
# ------------------------------------------------------------
def backfill_coords_img(layer):
    raster_layer = get_top_raster_layer()
    if not raster_layer:
        print("Warning: No raster layer found for backfilling COORDS_IMG")
        return
    
    layer.startEditing()
    updated_count = 0
    
    for feat in layer.getFeatures():
        coords_img = feat["COORDS_IMG"]
        needs_update = (
            coords_img is None or 
            str(coords_img).strip() == "" or 
            str(coords_img).upper() in ("NULL", "NONE") or
            coords_img in ("N/A", "NO_RASTER")
        )
        
        if needs_update:
            geom = feat.geometry()
            if geom and geom.isGeosValid() and not geom.isMultipart():
                polygon_points = geom.asPolygon()[0]
                coords_img_str = geo_to_pixel_coords(raster_layer, polygon_points)
                
                if coords_img_str:
                    idx = layer.fields().indexFromName("COORDS_IMG")
                    layer.changeAttributeValue(feat.id(), idx, coords_img_str)
                    updated_count += 1
                    print(f"Backfilled COORDS_IMG for feature {feat.id()}: {coords_img_str}")
    
    if layer.commitChanges():
        layer.triggerRepaint()
        if updated_count > 0:
            print(f"✓ Backfilled COORDS_IMG for {updated_count} existing features")
        else:
            print("No features needed COORDS_IMG backfilling")
    else:
        print("Error committing changes")
        layer.rollBack()

backfill_coords_img(annotation_layer)

# ------------------------------------------------------------
# ANNOTATION TOOL
# ------------------------------------------------------------
class AnnotationTool(QgsMapTool):
    def __init__(self, canvas, layer):
        super().__init__(canvas)
        self.canvas = canvas
        self.layer = layer
        self.start_point = None
        self.rubber = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        self.rubber.setWidth(2)
        self.current_class = DEFAULT_CLASS
        self.set_rubber_color()

        self.text_item = QGraphicsSimpleTextItem()
        self.text_item.setZValue(1000)
        self.text_item.setVisible(False)
        self.canvas.scene().addItem(self.text_item)

    def set_class(self, cls):
        self.current_class = cls
        self.set_rubber_color()
        if cls == "POSITIVE":
            pos_button.setStyleSheet("background-color: green; color: white; font-weight: bold")
            neg_button.setStyleSheet("")
        else:
            neg_button.setStyleSheet("background-color: red; color: white; font-weight: bold")
            pos_button.setStyleSheet("")

    def set_rubber_color(self):
        if self.current_class == "POSITIVE":
            self.rubber.setColor(QColor(0, 255, 0, 100))
        else:
            self.rubber.setColor(QColor(255, 0, 0, 100))

    def update_text_item(self, point):
        scene_point = self.canvas.getCoordinateTransform().transform(point)
        self.text_item.setPos(QPointF(scene_point.x(), scene_point.y()))
        self.text_item.setText(self.current_class)
        self.text_item.setVisible(True)

    def canvasPressEvent(self, e):
        self.start_point = self.toMapCoordinates(e.pos())
        self.rubber.reset(QgsWkbTypes.PolygonGeometry)
        self.update_text_item(self.start_point)

    def canvasMoveEvent(self, e):
        if not self.start_point:
            return
        end_point = self.toMapCoordinates(e.pos())
        rect = QgsRectangle(self.start_point, end_point)
        geom = QgsGeometry.fromRect(rect)
        self.rubber.reset(QgsWkbTypes.PolygonGeometry)
        self.rubber.addGeometry(geom, None)
        top_center = rect.center()
        self.update_text_item(top_center)

    def canvasReleaseEvent(self, e):
        if not self.start_point:
            return
        end_point = self.toMapCoordinates(e.pos())
        rect = QgsRectangle(self.start_point, end_point)
        geom = QgsGeometry.fromRect(rect)

        if geom.isEmpty():
            print("Empty geometry, skipping")
            return

        src_img = get_top_raster_filename()
        raster_layer = get_top_raster_layer()
        
        feat = QgsFeature(self.layer.fields())
        feat.setGeometry(geom)
        feat["CLASS"] = self.current_class
        feat["SRC_IMAGE"] = src_img

        coords_str = ""
        coords_img_str = ""
        if geom.isGeosValid() and not geom.isMultipart():
            polygon_points = geom.asPolygon()[0]
            coords_str = ";".join([f"{p.x():.6f},{p.y():.6f}" for p in polygon_points])
            feat["COORDS"] = coords_str
            
            if raster_layer:
                coords_img_str = geo_to_pixel_coords(raster_layer, polygon_points)
                if coords_img_str:
                    feat["COORDS_IMG"] = coords_img_str
                    print(f"Image coords: {coords_img_str}")
                else:
                    feat["COORDS_IMG"] = "N/A"
            else:
                feat["COORDS_IMG"] = "NO_RASTER"

        self.layer.dataProvider().addFeatures([feat])
        self.layer.updateExtents()
        
        apply_color_symbology(self.layer)
        self.layer.triggerRepaint()
        self.canvas.refresh()

        print(f"Added {self.current_class} polygon from {src_img}")
        self.start_point = None
        self.text_item.setVisible(False)

# ------------------------------------------------------------
# CREATE BUTTONS
# ------------------------------------------------------------
canvas = iface.mapCanvas()

annot_button = QPushButton("Annotation: OFF")
annot_button.setStyleSheet("background-color: yellow")

pos_button = QPushButton("POSITIVE")
neg_button = QPushButton("NEGATIVE")

widget = QWidget()
layout = QHBoxLayout()
layout.setContentsMargins(0,0,0,0)
layout.addWidget(annot_button)
layout.addWidget(pos_button)
layout.addWidget(neg_button)
widget.setLayout(layout)
iface.addToolBarWidget(widget)
iface.annotation_button_widget = widget

annotation_tool = AnnotationTool(canvas, annotation_layer)
annotating = False

def toggle_annotation():
    global annotating
    annotating = not annotating
    if annotating:
        canvas.setMapTool(annotation_tool)
        annot_button.setText("Annotation: ON")
        annot_button.setStyleSheet("background-color: lightblue")
        print("Annotation mode ON")
    else:
        canvas.unsetMapTool(annotation_tool)
        annot_button.setText("Annotation: OFF")
        annot_button.setStyleSheet("background-color: yellow")
        print("Annotation mode OFF")

def set_positive():
    annotation_tool.set_class("POSITIVE")

def set_negative():
    annotation_tool.set_class("NEGATIVE")

annot_button.clicked.connect(toggle_annotation)
pos_button.clicked.connect(set_positive)
neg_button.clicked.connect(set_negative)

# Set default class styling
annotation_tool.set_class(DEFAULT_CLASS)

print("Annotation tool loaded with color coding. Output shapefile:", OUTPUT_SHP)
print("POSITIVE = Green, NEGATIVE = Red")

