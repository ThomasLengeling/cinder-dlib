#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

using namespace ci;
using namespace ci::app;
using namespace std;

class ClasificationApp : public App {
  public:
	void setup() override;
	void mouseDown( MouseEvent event ) override;
	void update() override;
	void draw() override;
};

void ClasificationApp::setup()
{
}

void ClasificationApp::mouseDown( MouseEvent event )
{
}

void ClasificationApp::update()
{
}

void ClasificationApp::draw()
{
	gl::clear( Color( 0, 0, 0 ) ); 
}

CINDER_APP( ClasificationApp, RendererGl )
