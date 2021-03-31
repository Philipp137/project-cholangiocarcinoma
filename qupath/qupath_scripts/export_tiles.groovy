
import ij.*
import java.awt.Color
import java.awt.image.BufferedImage
import javax.imageio.ImageIO;
import qupath.lib.images.servers.LabeledImageServer
import qupath.imagej.detect.tissue.PositivePixelCounterIJ

def imageData = getCurrentImageData()


// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'tiles')
mkdirs(pathOutput)

// Define output resolution
double requestedPixelSize = 1.0

// Convert to downsample
double downsample = 4//requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()

static double tile_white_fraction(img, width)
{       
        //new ImagePlus("Image", img).show() // prints the image
        int white_pixel = 0
        int num_pixel = 0
        for (int i = 0; i < width; i++) {
                for (int j = 0; j < width; j++) {
                    Color mycolor = new Color(img.getRGB(i,j));
                    num_pixel ++
                    if ((mycolor.getRed() + mycolor.getRed() + mycolor.getRed()) >(3*235) ){ white_pixel ++}
            }
        }

        return white_pixel/num_pixel
}       

def server = imageData.getServer()
def filename = server.getMetadata().getName()
def pathology_number = filename.split('.tif')[0]
// find smallest x and y coordinate of all tiles
// and check if all tiles have the same width
int i1 = 0
for (annotation in getAnnotationObjects()) {
    roi = annotation.getROI()
    def tile = RegionRequest.createInstance(imageData.getServerPath(),downsample,roi)
    String tiletype = annotation.getParent().getPathClass()
//    if (!tiletype.equals("Tumor")){ continue } // continue to next tile if it is not a tumor tile
    if (!roi.getRoiName().equals("Rectangle")){ continue }
    if (i1 >0){
        if(tile.x < min_x){ min_x = tile.x }
        if(tile.y < min_y){ min_y = tile.y }
    } else{
        min_x = tile.x
        min_y = tile.y
        width = tile.width        
    }
    if (width != tile.width){ 
        print "!!!! all tiles need to have same size (width)!!!!"+width + tile.width+ annotation.getName()
        width = 0
        break
    }
    i1++
}


//runPlugin('qupath.imagej.detect.tissue.PositivePixelCounterIJ', '{"downsampleFactor": 4,  "gaussianSigmaMicrons": 2.0,  "thresholdStain1": 9.0,  "thresholdStain2": 0.1,  "addSummaryMeasurements": true,  "clearParentMeasurements": true,  "appendDetectionParameters": false,  "legacyMeasurements0.1.2": false}');
// now loop through all tumor tiles and export them into the desired directory
int i2 = 0
println "Exporting Tiles: "
for (annotation in getAnnotationObjects()) {
    String tiletype = annotation.getParent().getPathClass()
    //if (!tiletype.equals("Tumor")){ continue } // continue to next tile if it is not a tumor tile
    //runPlugin('qupath.imagej.detect.tissue.PositivePixelCounterIJ', '{"downsampleFactor": 1,  "gaussianSigmaMicrons": 2.0,  "thresholdStain1": 9.0,  "thresholdStain2": 0.1,  "addSummaryMeasurements": true,  "clearParentMeasurements": true,  "appendDetectionParameters": false,  "legacyMeasurements0.1.2": false}');
    
    // get region of interest in this case its an rectangle in the tumor part
    roi = annotation.getROI()
    // tile_name is something like "Tile 245"
    tile_name = annotation.getName()
    if (!roi.getRoiName().equals("Rectangle")){ continue }
    // Tile is just a rectangle 
    def tile = RegionRequest.createInstance(imageData.getServerPath(),downsample,roi)
    def img = server.readBufferedImage(tile)
    // calculate the fraction of white pixels in the current tile and exclude it if the number of
    // pixels is larger then 60% 
    if (tile_white_fraction(img, width/downsample)>0.6){
        //new ImagePlus("Image", img).show() // prints the image
        print " "
        print tile_name +" is too white   ("+tile_white_fraction(img, width/downsample) + "% )"
        println " "
        continue
    }
    // compute relative coordinates of the tile with respect to upper left corner of all tiles
    int x_rel = (tile.x-min_x)/width
    int y_rel = (tile.y-min_y)/width
    String tilename = String.format("%s_%s_%d_%d.png",pathology_number,tile_name.split(' ')[1], x_rel, y_rel)
    println tilename  
    def outputPath = buildFilePath(pathOutput, tilename)
    writeImageRegion(server, tile, outputPath)
    i2++
    
    //print("wrote " + filename)
}

// finally we do some garbage collection in order
// to not kill our selfs
Thread.sleep(100)
// Try to reclaim whatever memory we can, including emptying the tile cache
javafx.application.Platform.runLater {
    getCurrentViewer().getImageRegionStore().cache.clear()
    System.gc() // gc for garbage collection<
}
Thread.sleep(100)


print 'done!'





/// qupath commands:
// roi.getCentroidX() ... center of rectangular (x position)
// roi.getLength() ... length of roi?



