#include "Qts/viewqt.h"
#include "Qts/modelsqt.h"
#include "Qts/streamthread.h"

#include <iostream>
#include <stdio.h>

#include <QPainter>
#include <QBrush>
#include <QPixmap>
#include <cmath>
#include <QGraphicsSceneEvent>
#include <QMimeData>
#include <QByteArray>
#include <QFont>
char viewstrbuff[200];
QPointF points[100];

void DefaultScene::mousePressEvent ( QGraphicsSceneMouseEvent * event )
{
    emit clicked(event);
}
void DefaultScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    QPen pen;
    QFont txtfont("Roman",40);
    txtfont.setBold(true);
    pen.setColor(QColor(255,255,255));
    pen.setCapStyle(Qt::RoundCap);
    pen.setJoinStyle(Qt::RoundJoin);
    pen.setWidth(10);
    painter->setPen(QColor(243,134,48,150));
    painter->setFont(txtfont);
    painter->drawText(rect, Qt::AlignCenter,"打开文件\nOpen File");
}
TrkScene::TrkScene(const QRectF & sceneRect, QObject * parent):QGraphicsScene(sceneRect, parent)
{
    streamThd=NULL;
}
TrkScene::TrkScene(qreal x, qreal y, qreal width, qreal height, QObject * parent):QGraphicsScene( x, y, width, height, parent)
{
    streamThd=NULL;
}
void TrkScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    //debuggingFile<<streamThd->inited<<std::endl;
    if(streamThd!=NULL&&streamThd->inited)
    {
        updateFptr(streamThd->frameptr, streamThd->frameidx);
    }
    painter->setBrush(bgBrush);
    painter->drawRect(rect);
//    painter->setBrush(QColor(0,0,0,100));
//    painter->drawRect(rect);
    painter->setBrush(Qt::NoBrush);
	if (streamThd != NULL&&streamThd->inited)
	{
		int nFeatures = streamThd->tracker->nFeatures;
		int nSearch = streamThd->tracker->nSearch;
		float2* prePts = (float2*)streamThd->tracker->prePts.data;
		float2* nextPts = (float2*)streamThd->tracker->nextPts.data;
		unsigned char* status = streamThd->tracker->status->cpu_ptr();
		MemBuff<int2>& corners = *(streamThd->tracker->corners);
		Tracks& tracks = *streamThd->tracker->tracksGPU;
		int* lenVec = tracks.lenData->cpu_ptr();
		int acceptPtrNum = streamThd->tracker->acceptPtrNum;
		int* nbCount = streamThd->tracker->nbCount->cpu_ptr();
		linepen.setColor(QColor(255, 200, 200));
		linepen.setWidth(3);
		painter->setPen(linepen);
		painter->setFont(QFont("System", 20, 2));
		QString infoString = "fps:" + QString::number(streamThd->fps) + "\n"
			+ "acceptPtrNum:" + QString::number(acceptPtrNum) + "\n"
			+ "thresh:" + QString::number(thresh) + "\n";
		painter->drawText(rect, Qt::AlignLeft | Qt::AlignTop, infoString);
		painter->setFont(QFont("System", 20, 2));
		/*
		linepen.setColor(Qt::red);
		linepen.setWidth(1);
		painter->setPen(linepen);
		
		for (int i = 0; i < acceptPtrNum; i++)
		{
			int x = corners[i].x, y = corners[i].y;
			painter->drawLine(x - 10,y,x + 10 , y);
			painter->drawLine(x , y- 10, x , y + 10);
		}
		*/
		linepen.setColor(QColor(0,255,0));
		linepen.setWidth(2);
		painter->setFont(QFont("System", 10, 0));
		painter->setPen(linepen);
		for (int i = 0; i < nFeatures; i++)
		{
			int len = lenVec[i];
			if (len > 0)
			{
				FeatPts curPt = *tracks.getPtr(i);
				int alhpa = len;
				UperLowerBound(alhpa, 0, 255);
				linepen.setWidth(2);
				linepen.setColor(QColor(0, 255, 0, alhpa));
				painter->setPen(linepen);
				painter->drawPoint(curPt.x,curPt.y);
				/*
				for (int j = i+1; j < nFeatures; j++)
				{
					int val = nbCount[i*nFeatures + j];
					if (lenVec[j]>0&&val > 10)
					{
						FeatPts Pt = *tracks.getPtr(j);
						int alhpa = val;
						UperLowerBound(alhpa, 0, 255);
						linepen.setWidth(1);
						linepen.setColor(QColor(0, 255, 0, alhpa));
						painter->setPen(linepen);
						painter->drawPoint(curPt.x, curPt.y);
						painter->drawLine(curPt.x, curPt.y, Pt.x,Pt.y);
					}
				}
				*/
			}
		}
		
	}
    //update();
    //views().at(0)->update();
}
void TrkScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	/*
    if(event->button()==Qt::RightButton)
    {
        int x = event->scenePos().x(),y=event->scenePos().y();
        DragBBox* newbb = new DragBBox(x-10,y-10,x+10,y+10);
        int pid = dragbbvec.size();
        newbb->bbid=pid;
        newbb->setClr(255,255,255);
        sprintf(newbb->txt,"%c\0",pid+'A');
        dragbbvec.push_back(newbb);
        addItem(newbb);
    }
    QGraphicsScene::mousePressEvent(event);
	*/
}
void TrkScene::updateFptr(unsigned char * fptr,int fidx)
{
    bgBrush.setTextureImage(QImage(fptr,streamThd->framewidth,streamThd->frameheight,QImage::Format_RGB888));
    frameidx=fidx;
    //debuggingFile<<frameidx<<std::endl;
}
void TrkScene::clear()
{
    bgBrush.setStyle(Qt::NoBrush);
}
