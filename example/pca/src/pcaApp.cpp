#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

using namespace ci;
using namespace ci::app;
using namespace std;

class pcaApp : public App {
  public:
	void setup() override;
	void mouseDown( MouseEvent event ) override;
	void update() override;
	void draw() override;
};

void pcaApp::setup()
{
}

void pcaApp::mouseDown( MouseEvent event )
{
}

void pcaApp::update()
{
}

void pcaApp::draw()
{
	gl::clear( Color( 0, 0, 0 ) ); 
}

CINDER_APP( pcaApp, RendererGl )
