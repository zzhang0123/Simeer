1. Current limtod generate simulations using healpix beam; but meerklass beam is in difference format. I am developping this package to make an optimal package for MeerKLASS TOD simulation.
It can use understood as an optimal implementation of the generate_sky_TOD for meerklass beam; all other elements unchange. 

你可以用两个package里的API；或者读懂他们有助于你写这个package。一个是limtod：/Users/zzhang/Workspace/RadioCosmology/limTOD  另一个是TIBEC：/Users/zzhang/Workspace/RadioCosmology/TIBEC

我有如下的两种方案：1. 将beam coordinates映射到sky，然后interpolate sky； 2. 将sky coords映射到beam，然后interpolate beam。 我本倾向于担心前者，因为可以直接使用healpy内置的interpolation功能，但它不能track commonly defined的sky pixels，也许可以work around？至于后者，我刚意识到beam file有三十几G。。估计interpolate beam不如interpolate sky。

以下是我的笔记（不是作为source of truth；只是作为background；可能有对有错）：

I created a beam module to read and interpolate the MeerKAT holographic beam both in angle and frequency. See attached. It's part the the museek point source calibration branch

beam will be in local folder; I'm still downloading it.

I am considering an optimal strategy to avoid full sky beam. The sky model is full sky (and usually in healpix). I think the optimal strategy is to map the beam angular coordinates to celestial sky coordinates. Then interpolate the sky map on the transformed coordinate grids and multiply with beam afterwards. So beam is not rotated, but the sky model is interpolated.

(This simple coordinate mapping will work for Stokes I. For Q/U, extra transformation of fields is needed.)

The native limTOD implementation did not use this strategy because it was designed to explicitly track sky pixels for map-making. However, for the MeerKAT/MeerKLASS simulations here, we can relax that constraint and instead use a more optimal simulation strategy.

Q: I wonder if transforming sky to beam coordinates and interpolate the beam would be more efficient? currently in my code interpolation happens in the beam image. The other point is if we can use the fact that points close in time should be similar and interpolate there (e.g. if the healpix resolution is much higher than the fwhm). I confess I didn't think deeply about it. At the moment I am just using the beam for point sources. But I assume convolving over sky and frequency cube can be slow. What about FFTs?

A: Transforming the (celestial) sky into beam coordinates is also viable — this is essentially the current approach in limtod:
Pros: we can consistently track the same sky parameterization across time.
Cons: we need to transform the full sky coordinate set, unless we pre-select the subset of sky coordinates actually involved.
That said, this “con” is not a big deal — it is just a moderate computational overhead.
Transforming the beam into sky coordinates is the opposite: we only need to transform a small number of beam coordinates, but we lose a unified sky pixelization/parameterization.
It may still be preferable to go with the first approach, since it integrates more naturally with our Bayesian map-making framework.
We have a 3D regular grid for the beam, I think?  Two dimensions correspond to angular direction and one dimension to frequency. For each new 2D directional grid (transformed sky map coordinates), we want to interpolate the beam response while the full frequency coordinates does not change.
This is usually not too heavy in Python if treated as a batch of vectorised 2D interpolations rather than a true 3D interpolation. Since the frequency axis is unchanged, the main trick is to precompute the angular interpolation indices/weights once and then apply them vectorially to all frequencies. This is typically much faster than repeatedly calling a generic interpolator.
I would lean against using FFTs, mainly because of concerns about the coupling between the large survey area (curvature effects), beam asymmetry, and possible non-zero net spin relative to the sky-coordinate basis.
(That said, I’m not entirely sure. It may still be a sufficiently good approximation in practice. But we would need some ground-truth validation to assess that properly.)

Well, the sky frequency is also higher than the beam. So some interpolation is needed there. Though it is small. I guess we can go ahead with the brute force approach for now and see how ir goes. My idea of going from sky to beam was to pre select the sky radius to convolve with. Say 6 degrees. Of course, if the radius is very large, you might as well use spherical harmonics. But
the measured beam is only 6 deg anyway. But I'm sure you have gave more thoughts about this than me. The beam module I sent should help.