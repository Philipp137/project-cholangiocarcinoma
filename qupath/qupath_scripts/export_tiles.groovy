import static qupath.lib.gui.scripting.QPEx.*

MY_DATA_DIR = '/home/phil/develop/python/project-cholangiocarcinoma/data/project-cholangiocarcinoma/CCC/pictures/'
// Get the current image (supports 'Run for project')
def imageData = getCurrentImageData()

// Define output path (here, relative to project)
//def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(MY_DATA_DIR)
print pathOutput
mkdirs(pathOutput)

// Define output resolution in calibrated units (e.g. µm if available)
double requestedPixelSize = 5.0

// Convert output resolution to a downsample factor
double pixelSize = imageData.getServer().getPixelCalibration().getAveragedPixelSize()
double downsample = requestedPixelSize / pixelSize

// Create an exporter that requests corresponding tiles from the original & labelled image servers
def tileexp = new TileExporter(imageData)
    .downsample(downsample)   // Define export resolution
    .imageExtension('.png')   // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .tileSize(512)            // Define size of each tile, in pixels
    .annotatedTilesOnly(true) // If true, only export tiles if there is a (classified) annotation present
    .overlap(0)              // Define overlap, in pixel units at the export resolution
tileexp.writeTiles(pathOutput)   // Write tiles to the specified directory
print 'Done!'