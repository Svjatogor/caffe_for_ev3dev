ó
/ ¡Wc           @   só   d  d l  Z d  d l Z d  d l m Z d  d l m Z y d  d l m	 Z	 Wn- d  d l
 Z
 e
 j d k ru d GHq|   n Xe d  Z d d	  Z d
   Z d   Z d d  Z d   Z d d d     YZ e d  Z d d  Z d   Z d S(   iÿÿÿÿN(   t   zoom(   t   resize(   t	   caffe_pb2i   i    s3   Failed to include caffe_pb2, things might go wrong!c         C   s¢   | r t  j |  j  } n t  j |  j  } |  j d  si |  j d  si |  j d  si |  j d  r | j |  j |  j |  j |  j	  S| j |  j
 j  Sd S(   s   
    Convert a blob proto to an array. In default, we will just return the data,
    unless return_diff is True, in which case we will return the diff.
    t   numt   channelst   heightt   widthN(   t   npt   arrayt   difft   datat   HasFieldt   reshapeR   R   R   R   t   shapet   dim(   t   blobt   return_diffR
   (    (    s   /caffe/python/caffe/io.pyt   blobproto_to_array   s    <"c         C   sm   t  j   } | j j j |  j  | j j |  j t  j  | d k	 ri | j
 j | j t  j  n  | S(   sÌ   Converts a N-dimensional array to blob proto. If diff is given, also
    convert the diff. You need to make sure that arr and diff have the same
    shape, and this function does not do sanity check.
    N(   R   t	   BlobProtoR   R   t   extendR
   t   astypet   floatt   flatt   NoneR	   (   t   arrR	   R   (    (    s   /caffe/python/caffe/io.pyt   array_to_blobproto$   s    c         C   s?   t  j   } | j j g  |  D] } t |  ^ q  | j   S(   sx   Converts a list of arrays to a serialized blobprotovec, which could be
    then passed to a network for processing.
    (   R   t   BlobProtoVectort   blobsR   R   t   SerializeToString(   t	   arraylistt   vecR   (    (    s   /caffe/python/caffe/io.pyt    arraylist_to_blobprotovector_str1   s    )c         C   s9   t  j   } | j |   g  | j D] } t |  ^ q# S(   s<   Converts a serialized blobprotovec to a list of arrays.
    (   R   R   t   ParseFromStringR   R   (   t   strR   R   (    (    s   /caffe/python/caffe/io.pyt    blobprotovector_str_to_arraylist:   s    c         C   s   |  j  d k r t d   n  t j   } |  j \ | _ | _ | _ |  j t	 j
 k ri |  j   | _ n | j j |  j  | d k	 r | | _ n  | S(   s»   Converts a 3-dimensional array to datum. If the array has dtype uint8,
    the output data will be encoded as a string. Otherwise, the output data
    will be stored in float format.
    i   s   Incorrect array shape.N(   t   ndimt
   ValueErrorR   t   DatumR   R   R   R   t   dtypeR   t   uint8t   tostringR
   t
   float_dataR   R   R   t   label(   R   R*   t   datum(    (    s   /caffe/python/caffe/io.pyt   array_to_datumB   s    c         C   su   t  |  j  r@ t j |  j d t j j |  j |  j |  j  St j	 |  j
  j t  j |  j |  j |  j  Sd S(   s|   Converts a datum to an array. Note that the label is not returned,
    as one can easily get it by calling datum.label.
    R&   N(   t   lenR
   R   t
   fromstringR'   R   R   R   R   R   R)   R   R   (   R+   (    (    s   /caffe/python/caffe/io.pyt   datum_to_arrayT   s
    t   Transformerc           B   s_   e  Z d  Z d   Z d   Z d   Z d   Z d   Z d   Z d   Z	 d   Z
 d	   Z RS(
   s  
    Transform input for feeding into a Net.

    Note: this is mostly for illustrative purposes and it is likely better
    to define your own input preprocessing routine for your needs.

    Parameters
    ----------
    net : a Net for which the input should be prepared
    c         C   s:   | |  _  i  |  _ i  |  _ i  |  _ i  |  _ i  |  _ d  S(   N(   t   inputst	   transposet   channel_swapt	   raw_scalet   meant   input_scale(   t   selfR1   (    (    s   /caffe/python/caffe/io.pyt   __init__m   s    					c         C   s1   | |  j  k r- t d j | |  j     n  d  S(   Ns#   {} is not one of the net inputs: {}(   R1   t	   Exceptiont   format(   R7   t   in_(    (    s   /caffe/python/caffe/io.pyt   __check_inputu   s    	c   
      C   sP  |  j  |  | j t j d t } |  j j |  } |  j j |  } |  j j |  } |  j	 j |  } |  j
 j |  } |  j | d }	 | j d  |	 k rµ t | |	  } n  | d k	 rÓ | j |  } n  | d k	 r| | d d  d d  f } n  | d k	 r| | 9} n  | d k	 r3| | 8} n  | d k	 rL| | 9} n  | S(   sD  
        Format input for Caffe:
        - convert to single
        - resize to input dimensions (preserving number of channels)
        - transpose dimensions to K x H x W
        - reorder channels (for instance color to BGR)
        - scale raw input (e.g. from [0, 1] to [0, 255] for ImageNet models)
        - subtract mean
        - scale feature

        Parameters
        ----------
        in_ : name of input blob to preprocess for
        data : (H' x W' x K) ndarray

        Returns
        -------
        caffe_in : (K x H x W) ndarray for input to a Net
        t   copyi   N(   t   _Transformer__check_inputR   R   t   float32t   FalseR2   t   getR3   R4   R5   R6   R1   R   t   resize_imageR   (
   R7   R;   R
   t   caffe_inR2   R3   R4   R5   R6   t   in_dims(    (    s   /caffe/python/caffe/io.pyt
   preprocessz   s*    "c   	      C   s&  |  j  |  | j   j   } |  j j |  } |  j j |  } |  j j |  } |  j j |  } |  j j |  } | d k	 r | | :} n  | d k	 r« | | 7} n  | d k	 rÄ | | :} n  | d k	 rû | t
 j |  d d  d d  f } n  | d k	 r"| j t
 j |   } n  | S(   s<   
        Invert Caffe formatting; see preprocess().
        N(   R>   R=   t   squeezeR2   RA   R3   R4   R5   R6   R   R   t   argsort(	   R7   R;   R
   t   decaf_inR2   R3   R4   R5   R6   (    (    s   /caffe/python/caffe/io.pyt	   deprocess¤   s$    +c         C   sP   |  j  |  t |  t |  j |  d k r? t d   n  | |  j | <d S(   s  
        Set the input channel order for e.g. RGB to BGR conversion
        as needed for the reference ImageNet model.

        Parameters
        ----------
        in_ : which input to assign this channel order
        order : the order to transpose the dimensions
        i   sI   Transpose order needs to have the same number of dimensions as the input.N(   R>   R-   R1   R9   R2   (   R7   R;   t   order(    (    s   /caffe/python/caffe/io.pyt   set_transpose»   s    
#c         C   sJ   |  j  |  t |  |  j | d k r9 t d   n  | |  j | <d S(   s  
        Set the input channel order for e.g. RGB to BGR conversion
        as needed for the reference ImageNet model.
        N.B. this assumes the channels are the first dimension AFTER transpose.

        Parameters
        ----------
        in_ : which input to assign this channel order
        order : the order to take the channels.
            (2,1,0) maps RGB to BGR for example.
        i   sO   Channel swap needs to have the same number of dimensions as the input channels.N(   R>   R-   R1   R9   R3   (   R7   R;   RJ   (    (    s   /caffe/python/caffe/io.pyt   set_channel_swapË   s    c         C   s   |  j  |  | |  j | <d S(   s  
        Set the scale of raw features s.t. the input blob = input * scale.
        While Python represents images in [0, 1], certain Caffe models
        like CaffeNet and AlexNet represent images in [0, 255] so the raw_scale
        of these models must be 255.

        Parameters
        ----------
        in_ : which input to assign this scale factor
        scale : scale coefficient
        N(   R>   R4   (   R7   R;   t   scale(    (    s   /caffe/python/caffe/io.pyt   set_raw_scaleÝ   s    c         C   sè   |  j  |  | j } | j d k rq | d |  j | d k rO t d   n  | d d  t j t j f } nf t |  d k r d	 | } n  t |  d k r± t d   n  | |  j | d k r× t d   n  | |  j | <d S(
   sÙ   
        Set the mean to subtract for centering the data.

        Parameters
        ----------
        in_ : which input to assign this mean.
        mean : mean ndarray (input dimensional or broadcastable)
        i   i    s&   Mean channels incompatible with input.Ni   i   s   Mean shape invalids)   Mean shape incompatible with input shape.(   i   (	   R>   R   R#   R1   R$   R   t   newaxisR-   R5   (   R7   R;   R5   t   ms(    (    s   /caffe/python/caffe/io.pyt   set_meanì   s    		"c         C   s   |  j  |  | |  j | <d S(   sK  
        Set the scale of preprocessed inputs s.t. the blob = blob * scale.
        N.B. input_scale is done AFTER mean subtraction and other preprocessing
        while raw_scale is done BEFORE.

        Parameters
        ----------
        in_ : which input to assign this scale factor
        scale : scale coefficient
        N(   R>   R6   (   R7   R;   RM   (    (    s   /caffe/python/caffe/io.pyt   set_input_scale  s    (   t   __name__t
   __module__t   __doc__R8   R>   RE   RI   RK   RL   RN   RQ   RR   (    (    (    s   /caffe/python/caffe/io.pyR0   b   s   
			*					c         C   s¼   t  j t  j j |  d |  j t j  } | j d k r} | d d  d d  t j f } | r¸ t j	 | d  } q¸ n; | j
 d d k r¸ | d d  d d  d d  f } n  | S(   s´  
    Load an image converting from grayscale or alpha as needed.

    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).

    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    t   as_greyi   Ni   i   i   (   i   i   i   (   t   skimaget   img_as_floatt   iot   imreadR   R   R?   R#   RO   t   tileR   (   t   filenamet   colort   img(    (    s   /caffe/python/caffe/io.pyt
   load_image  s    ."(i   c   
      C   s#  |  j  d d k s& |  j  d d k rË |  j   |  j   } } | | k r |  | | | } t | | d | } | | | | } qt j | d | d |  j  d f d t j } | j |  | SnH t t j	 | d t
 t j	 |  j  d    }	 t |  |	 d d | } | j t j  S(	   s=  
    Resize an image array with interpolation.

    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.

    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    iÿÿÿÿi   i   RJ   i    R&   i   (   i   (   R   t   mint   maxR   R   t   emptyR?   t   fillt   tupleR   R   R    R   (
   t   imt   new_dimst   interp_ordert   im_mint   im_maxt   im_stdt   resized_stdt
   resized_imt   retRM   (    (    s   /caffe/python/caffe/io.pyRB   2  s    &$/c         C   sý  t  j |  d j  } t  j |  } | d  d } d | d | d f } d | d | d f } t  j d d t } d } xO | D]G } x> | D]6 }	 | |	 | | d |	 | d f | | <| d 7} q Wq Wt  j | d  t  j | d | d g  | d <t  j | d  } t  j d t |   | d | d | d	 f d t  j }
 d } x¢ |  D] } xQ | D]I } | | d | d  | d | d
  d d  f |
 | <| d 7} qhW|
 | d |  d d  d d d	  d d  f |
 | d | +q[W|
 S(   s3  
    Crop images into the four corners, center, and their mirrored versions.

    Parameters
    ----------
    image : iterable of (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.

    Returns
    -------
    crops : (10*N x H x W x K) ndarray of crops for number of inputs N.
    i    i   g       @i   i   i   R&   i
   iÿÿÿÿi   N(   i   i   (   i   i   (   i   i   (	   R   R   R   Rb   t   intR[   t   concatenateR-   R?   (   t   imagest	   crop_dimst   im_shapet	   im_centert	   h_indicest	   w_indicest   crops_ixt   currt   it   jt   cropst   ixRe   t   crop(    (    s   /caffe/python/caffe/io.pyt
   oversampleU  s0    &!9D(   i   i    (    (   t   numpyR   t
   skimage.ioRW   t   scipy.ndimageR    t   skimage.transformR   t   caffe.protoR   t   syst   version_infoR@   R   R   R   R   R"   R,   R/   R0   t   TrueR_   RB   R}   (    (    (    s   /caffe/python/caffe/io.pyt   <module>   s(   				µ#