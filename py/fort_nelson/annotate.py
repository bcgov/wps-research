'''20251209 qgis-based annotation.. multi-image 
'''

from qgis.PyQt.QtWidgets import QPushButton, QHBoxLayout, QWidget, QGraphicsSimpleTextItem
from qgis.PyQt.QtGui import QColor
from PyQt5.QtCore import QPointF, QVariant
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsField, QgsGeometry, QgsRectangle,
    QgsWkbTypes, QgsFeature, QgsMapLayer, QgsSymbol, QgsRendererCategory, QgsCategorizedSymbolRenderer, QgsFillSymbol
)
from qgis.gui import QgsMapTool, QgsRubberBand
import os

OUT_SHP="/ram/parker/annotation_labels.shp"
DEF_CLASS="NEGATIVE"

# Remove existing toolbar widget
if hasattr(iface,"annotation_button_widget"):
    try:
        p=iface.annotation_button_widget.parentWidget()
        if p and hasattr(p,"removeWidget"): p.removeWidget(iface.annotation_button_widget)
        iface.annotation_button_widget.setParent(None)
        iface.annotation_button_widget.deleteLater()
    except Exception as e: print("Remove widget failed:",e)
    finally: del iface.annotation_button_widget

# Create shapefile if missing
if not os.path.exists(OUT_SHP):
    vl=QgsVectorLayer("Polygon?crs=EPSG:4326","annotations","memory")
    pr=vl.dataProvider()
    pr.addAttributes([
        QgsField("CLASS",QVariant.String,"string",10),
        QgsField("SRC_IMAGE",QVariant.String,"string",200),
        QgsField("COORDS",QVariant.String,"string",2000),
        QgsField("COORDS_IMG",QVariant.String,"string",2000)
    ])
    vl.updateFields()
    from qgis.core import QgsVectorFileWriter
    QgsVectorFileWriter.writeAsVectorFormat(vl,OUT_SHP,"utf-8",driverName="ESRI Shapefile")

layer=QgsVectorLayer(OUT_SHP,"Annotations","ogr")
if not layer.isValid(): raise Exception("Shapefile load failed")
layer.setCrs(iface.mapCanvas().mapSettings().destinationCrs())
QgsProject.instance().addMapLayer(layer)

def ensure_fields(l):
    p=l.dataProvider(); fs=[f.name() for f in l.fields()]; added=False
    for n,t,lx in [("CLASS",QVariant.String,10),("SRC_IMAGE",QVariant.String,200),
                   ("COORDS",QVariant.String,2000),("COORDS_IMG",QVariant.String,2000)]:
        if n not in fs: p.addAttributes([QgsField(n,t,"string",lx)]); added=True
    if added: l.updateFields()
ensure_fields(layer)

def top_raster_at_point(pt):
    for lyr in QgsProject.instance().layerTreeRoot().layerOrder():
        if lyr.type()==QgsMapLayer.RasterLayer:
            node=QgsProject.instance().layerTreeRoot().findLayer(lyr.id())
            if node and node.isVisible() and lyr.extent().contains(pt): return lyr
    return None

def raster_fname_at_point(pt):
    lyr=top_raster_at_point(pt)
    return os.path.basename(lyr.source()) if lyr else "UNKNOWN"

def geo2pix_coords(raster,pts):
    if not raster or not raster.isValid(): return None
    e=raster.extent(); w,h=raster.width(),raster.height()
    pc=[]
    for p in pts:
        c=int((p.x()-e.xMinimum())/e.width()*w)
        r=int((e.yMaximum()-p.y())/e.height()*h)
        pc.append(f"{max(0,min(c,w-1))},{max(0,min(r,h-1))}")
    return ";".join(pc)

def apply_symbology(l):
    pos=QgsFillSymbol.createSimple({'color':'0,255,0,100','outline_color':'0,180,0,255','outline_width':'0.5'})
    neg=QgsFillSymbol.createSimple({'color':'255,0,0,100','outline_color':'180,0,0,255','outline_width':'0.5'})
    l.setRenderer(QgsCategorizedSymbolRenderer('CLASS',[
        QgsRendererCategory('POSITIVE',pos,'Positive'),
        QgsRendererCategory('NEGATIVE',neg,'Negative')]))
    l.triggerRepaint()
apply_symbology(layer)

class AnnotTool(QgsMapTool):
    def __init__(self,canvas,l):
        super().__init__(canvas); self.c=canvas; self.l=l
        self.start=None; self.r=QgsRubberBand(canvas,QgsWkbTypes.PolygonGeometry)
        self.r.setWidth(2); self.cls=DEF_CLASS
        self.text=QGraphicsSimpleTextItem(); self.text.setZValue(1000); self.text.setVisible(False)
        self.c.scene().addItem(self.text)

    def set_class(self,cls):
        self.cls=cls
        self.r.setColor(QColor(0,255,0,100) if cls=="POSITIVE" else QColor(255,0,0,100))
        pos_button.setStyleSheet("background-color: green; color:white; font-weight:bold" if cls=="POSITIVE" else "")
        neg_button.setStyleSheet("background-color: red; color:white; font-weight:bold" if cls=="NEGATIVE" else "")

    def update_text(self,pt):
        p=self.c.getCoordinateTransform().transform(pt)
        self.text.setPos(QPointF(p.x(),p.y())); self.text.setText(self.cls); self.text.setVisible(True)

    def canvasPressEvent(self,e):
        self.start=self.toMapCoordinates(e.pos()); self.r.reset(QgsWkbTypes.PolygonGeometry)
        self.update_text(self.start)

    def canvasMoveEvent(self,e):
        if not self.start: return
        end=self.toMapCoordinates(e.pos()); rect=QgsRectangle(self.start,end)
        self.r.reset(QgsWkbTypes.PolygonGeometry); self.r.addGeometry(QgsGeometry.fromRect(rect),None)
        self.update_text(rect.center())

    def canvasReleaseEvent(self,e):
        if not self.start: return
        end=self.toMapCoordinates(e.pos()); rect=QgsRectangle(self.start,end); geom=QgsGeometry.fromRect(rect)
        if geom.isEmpty(): return
        src=raster_fname_at_point(rect.center()); raster=top_raster_at_point(rect.center())
        feat=QgsFeature(self.l.fields()); feat.setGeometry(geom)
        feat["CLASS"]=self.cls; feat["SRC_IMAGE"]=src
        if geom.isGeosValid() and not geom.isMultipart():
            pts=geom.asPolygon()[0]; feat["COORDS"]=";".join([f"{p.x():.6f},{p.y():.6f}" for p in pts])
            feat["COORDS_IMG"]=geo2pix_coords(raster,pts) if raster else "NO_RASTER"
        self.l.dataProvider().addFeatures([feat]); self.l.updateExtents()
        apply_symbology(self.l); self.l.triggerRepaint(); self.c.refresh()
        print(f"Added {self.cls} polygon from {src}"); self.start=None; self.text.setVisible(False)

c=iface.mapCanvas()
annot_button,pos_button,neg_button=QPushButton("Annotation: OFF"),QPushButton("POSITIVE"),QPushButton("NEGATIVE")
widget=QWidget(); layout=QHBoxLayout(); layout.setContentsMargins(0,0,0,0)
for b in [annot_button,pos_button,neg_button]: layout.addWidget(b)
widget.setLayout(layout); iface.addToolBarWidget(widget); iface.annotation_button_widget=widget

tool=AnnotTool(c,layer); annotating=False; tool.set_class(DEF_CLASS)

def toggle_annot():
    global annotating
    annotating=not annotating
    if annotating: c.setMapTool(tool); annot_button.setText("Annotation: ON"); annot_button.setStyleSheet("background-color: lightblue"); print("Annotation mode ON")
    else: c.unsetMapTool(tool); annot_button.setText("Annotation: OFF"); annot_button.setStyleSheet("background-color: yellow"); print("Annotation mode OFF")
def set_pos(): tool.set_class("POSITIVE")
def set_neg(): tool.set_class("NEGATIVE")

annot_button.clicked.connect(toggle_annot); pos_button.clicked.connect(set_pos); neg_button.clicked.connect(set_neg)
print("Annotation tool loaded. Shapefile:",OUT_SHP,"POSITIVE=Green, NEGATIVE=Red")

