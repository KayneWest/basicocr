from ocrolib import psegutils,morph,sl,lstm
from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter
from pylab import *
from ocrolib.toplevel import *
import ocrolib 


def DSAVE(title,image):
    if not args.debug: return
    if type(image)==list:
        assert len(image)==3
        image = transpose(array(image),[1,2,0])
    fname = "_"+title+".png"
    print "debug",fname
    imsave(fname,image)

def compute_line_seeds(binary,bottom,top,colseps,scale,threshold=0.2,vscale=1.0):
    """Base on gradient maps, computes candidates for baselines
    and xheights.  Then, it marks the regions between the two
    as a line seed."""
    t = threshold
    vrange = int(vscale*scale)
    bmarked = maximum_filter(bottom==maximum_filter(bottom,(vrange,0)),(2,2))
    bmarked *= (bottom>t*amax(bottom)*t)*(1-colseps)
    tmarked = maximum_filter(top==maximum_filter(top,(vrange,0)),(2,2))
    tmarked *= (top>t*amax(top)*t/2)*(1-colseps)
    tmarked = maximum_filter(tmarked,(1,20))
    seeds = zeros(binary.shape,'i')
    delta = max(3,int(scale/2))
    for x in range(bmarked.shape[1]):
        transitions = sorted([(y,1) for y in find(bmarked[:,x])]+[(y,0) for y in find(tmarked[:,x])])[::-1]
        transitions += [(0,0)]
        for l in range(len(transitions)-1):
            y0,s0 = transitions[l]
            if s0==0: continue
            seeds[y0-delta:y0,x] = 1
            y1,s1 = transitions[l+1]
            if s1==0 and (y0-y1)<5*scale: seeds[y1:y0,x] = 1
    seeds = maximum_filter(seeds,(1,int(1+scale)))
    seeds *= (1-colseps)
    #DSAVE("lineseeds",[seeds,0.3*tmarked+0.7*bmarked,binary])
    seeds,_ = morph.label(seeds)
    return seeds

def compute_gradmaps(binary,scale,vscale=1.0,hscale=1.0,usegauss=False):
    # use gradient filtering to find baselines
    boxmap = psegutils.compute_boxmap(binary,scale)
    cleaned = boxmap*binary
    #DSAVE("cleaned",cleaned)
    if usegauss:
        # this uses Gaussians
        grad = gaussian_filter(1.0*cleaned,(vscale*0.3*scale,
                                            hscale*6*scale),order=(1,0))
    else:
        # this uses non-Gaussian oriented filters
        grad = gaussian_filter(1.0*cleaned,(max(4,vscale*0.3*scale),
                                        hscale*scale),order=(1,0))
        grad = uniform_filter(grad,(vscale,hscale*6*scale))
    bottom = ocrolib.norm_max((grad<0)*(-grad))
    top = ocrolib.norm_max((grad>0)*grad)
    return bottom,top,boxmap



def compute_colseps_conv(binary,scale=1.0,csminheight=10,maxcolseps=2):
    """Find column separators by convolution and
    thresholding."""
    h,w = binary.shape
    # find vertical whitespace by thresholding
    smoothed = gaussian_filter(1.0*binary,(scale,scale*0.5))
    smoothed = uniform_filter(smoothed,(5.0*scale,1))
    thresh = (smoothed<amax(smoothed)*0.1)
    #DSAVE("1thresh",thresh)
    # find column edges by filtering
    grad = gaussian_filter(1.0*binary,(scale,scale*0.5),order=(0,1))
    grad = uniform_filter(grad,(10.0*scale,1))
    # grad = abs(grad) # use this for finding both edges
    grad = (grad>0.5*amax(grad))
    #DSAVE("2grad",grad)
    # combine edges and whitespace
    seps = minimum(thresh,maximum_filter(grad,(int(scale),int(5*scale))))
    seps = maximum_filter(seps,(int(2*scale),1))
    #DSAVE("3seps",seps)
    # select only the biggest column separators
    seps = morph.select_regions(seps,sl.dim0,min=csminheight*scale,nbest=maxcolseps+1)
    #DSAVE("4seps",seps)
    return seps


def compute_colseps(binary,scale,blackseps=True):
    """Computes column separators either from vertical black lines or whitespace."""
    colseps = compute_colseps_conv(binary,scale)
    #DSAVE("colwsseps",0.7*colseps+0.3*binary)
    if blackseps:
        seps = compute_separators_morph(binary,scale)
        #DSAVE("colseps",0.7*seps+0.3*binary)
        #colseps = compute_colseps_morph(binary,scale)
        colseps = maximum(colseps,seps)
        binary = minimum(binary,1-seps)
    return colseps,binary

def remove_hlines(binary,scale,maxsize=10):
    labels,_ = morph.label(binary)
    objects = morph.find_objects(labels)
    for i,b in enumerate(objects):
        if sl.width(b)>maxsize*scale:
            labels[b][labels[b]==i+1] = 0
    return array(labels!=0,'B')

def compute_segmentation(binary,scale):
    """Given a binary image, compute a complete segmentation into
    lines, computing both columns and text lines."""
    binary = array(binary,'B')

    # start by removing horizontal black lines, which only
    # interfere with the rest of the page segmentation
    binary = remove_hlines(binary,scale)

    # do the column finding
    colseps,binary = compute_colseps(binary,scale)

    # now compute the text line seeds
    bottom,top,boxmap = compute_gradmaps(binary,scale)
    seeds = compute_line_seeds(binary,bottom,top,colseps,scale)
    #DSAVE("seeds",[bottom,top,boxmap])

    # spread the text line seeds to all the remaining
    # components
    llabels = morph.propagate_labels(boxmap,seeds,conflict=0)
    spread = morph.spread_labels(seeds,maxdist=scale)
    llabels = where(llabels>0,llabels,spread*binary)
    segmentation = llabels*binary
    return segmentation

def compute_separators_morph(binary,scale,sepwiden=10,maxseps=2):
    """Finds vertical black lines corresponding to column separators."""
    d0 = int(max(5,scale/4))
    d1 = int(max(5,scale))+sepwiden
    thick = morph.r_dilation(binary,(d0,d1))
    vert = morph.rb_opening(thick,(10*scale,1))
    vert = morph.r_erosion(vert,(d0//2,sepwiden))
    vert = morph.select_regions(vert,sl.dim1,min=3,nbest=2*maxseps)
    vert = morph.select_regions(vert,sl.dim0,min=20*scale,nbest=maxseps)
    return vert

def compute_lines(segmentation,scale):
    """Given a line segmentation map, computes a list
    of tuples consisting of 2D slices and masked images."""
    lobjects = morph.find_objects(segmentation)
    lines = []
    for i,o in enumerate(lobjects):
        if o is None: continue
        if sl.dim1(o)<2*scale or sl.dim0(o)<scale: continue
        mask = (segmentation[o]==i+1)
        if amax(mask)==0: continue
        result = record()
        result.label = i+1
        result.bounds = o
        result.mask = mask
        lines.append(result)
    return lines



class Prediction(object):
    # TODO build a better model using Theano. Numpy is way to slow.
    def __init__(self,image,model="/Users/mkrzus/github/ocr_tools/ocropy/models/my-model3-00160000.pyrnn.gz"):
        self.image = image
        self.image_files = []
        self.segmentation = 0
        self.network = ocrolib.load_object(model,verbose=1)
        for x in self.network.walk(): x.postLoad()
        for x in self.network.walk(): 
          if isinstance(x,lstm.LSTM):
              x.allocate(5000)
        self.lnorm = getattr(self.network,"lnorm",None)

        #if height>0:
        #    lnorm.setHeight(-1)


    def midrange(self,image,frac=0.5):
        """Computes the center of the range of image values
        (for quick thresholding)."""
        return frac*(amin(image)+amax(image))

    def make_seg_white(self,image):
        image = image.copy()
        image[image==0] = 0xffffff
        return image

    def int2rgb(self,image):
        """Converts a rank 3 array with RGB values stored in the
        last axis into a rank 2 array containing 32 bit RGB values."""
        assert image.ndim==2
        a = zeros(list(image.shape)+[3],'B')
        a[:,:,0] = (image>>16)
        a[:,:,1] = (image>>8)
        a[:,:,2] = image
        return a

    def write_page_segmentation(self,image):
        """Writes a page segmentation, that is an RGB image whose values
        encode the segmentation of a page."""
        assert image.ndim==2
        assert image.dtype in [dtype('int32'),dtype('int64')]
        self.segmentation = self.int2rgb(self.make_seg_white(image))

    def write_image_binary(self,fname,image,verbose=0):
        """Return a binary image to class. This verifies first that the given image
        is, in fact, binary.  The image may be of any type, but must consist of only
        two values."""
        if verbose: print "# writing",fname
        assert image.ndim==2
        image = array(255*(image>self.midrange(image)),'B')
        return (fname, image)

    def isfloatarray(self,a):
        return a.dtype in [dtype('f'),dtype('float32'),dtype('float64')]

    def read_image_gray(self, fname):# ,pageno=0):
        """Read an image and returns it as a floating point array.
        The optional page number allows images from files containing multiple
        images to be addressed.  Byte and short arrays are rescaled to
        the range 0...1 (unsigned) or -1...1 (signed)."""
        #if type(fname)==tuple: fname,pageno = fname
        #assert pageno==0
        a = fname 
        if a.dtype==dtype('uint8'):
            a = a/255.0
        if a.dtype==dtype('int8'):
            a = a/127.0
        elif a.dtype==dtype('uint16'):
            a = a/65536.0
        elif a.dtype==dtype('int16'):
            a = a/32767.0
        elif self.isfloatarray(a):
            pass
        else:
            raise OcropusException("unknown image type: "+a.dtype)
        if a.ndim==3:
            a = mean(a,2)
        return a

    def prepare_line(self,line,pad=16):
        """Prepare a line for recognition; this inverts it, transposes
        it, and pads it."""
        line = line * 1.0/amax(line)
        line = amax(line)-line
        line = line.T
        if pad>0:
            w = line.shape[1]
            line = vstack([zeros((pad,w)),line,zeros((pad,w))])
        return line

    def extract(self):
        print('extracting information')
        binary = ocrolib.read_image_binary(self.image)
        binary = 1-binary
        scale = psegutils.estimate_scale(binary)
        segmentation = compute_segmentation(binary,scale)
        # ...lines = compute_lines(segmentation,scale)
        # compute the reading order
        lines = psegutils.compute_lines(segmentation,scale)
        order = psegutils.reading_order([l.bounds for l in lines])
        lsort = psegutils.topsort(order)
        # renumber the labels so that they conform to the specs
        nlabels = amax(segmentation)+1
        renumber = zeros(nlabels,'i')
        for i,v in enumerate(lsort): renumber[lines[v].label] = 0x010000+(i+1)
        segmentation = renumber[segmentation]
        lines = [lines[i] for i in lsort]
        #ocrolib.write_page_segmentation("%s.pseg.png"%outputdir,segmentation)
        # save the segmentation array
        self.write_page_segmentation(segmentation)
        cleaned = ocrolib.remove_noise(binary,8)
        for i,l in enumerate(lines):
            binline = psegutils.extract_masked(1-cleaned,l,pad=3,expand=3)
            #self.image_files.append(self.write_image_binary("%s/01%04x.bin.png"%("temp",i+1),binline))
            self.image_files.append(self.write_image_binary("01%04x.bin.png"%(i+1),binline))
    def predict_one(self, fname):
        # makes sure image is gray
        line = self.read_image_gray(fname)
        temp = amax(line)-line
        temp = temp*1.0/amax(temp)
        self.lnorm.measure(temp)
        line = self.lnorm.normalize(line,cval=amax(line))
        raw_line = line.copy()
        line = self.prepare_line(line,0)
        pred = self.network.predictString(line)
        print(pred)
        return pred

    def image_to_string(self):
        self.repository = []
        self.extract()
        for image in self.image_files:
            self.repository.append((self.predict_one(image[1]),image))




