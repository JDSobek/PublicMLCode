import math


def line_box_clipper3d(xmin, ymin, zmin, xmax, ymax, zmax, xin, yin, zin, vx, vy, vz):
    """
    Calculates outgoing point of a cube based on an incident point and a direction of travel.
    xmin, ymin, zmin specify respective minimum dimension of box
    xmax, ymax, zmax specify respective maximum dimension of box
    xin, yin, zin specify respective incident point coordinates
    vx, vy, vz specify direction of travel in respective coordinate
    """

    # figure out which faces the line is moving towards
    if vx >= 0.:
        distx = xmax - xin
    else:
        distx = xin - xmin

    if vy >= 0.:
        disty = ymax - yin
    else:
        disty = yin - ymin

    if vz >= 0.:
        distz = zmax - zin
    else:
        distz = zin - zmin

    # make sure the vector actually intercept the cube a second time
    if distx <= 0. or disty <= 0. or distz <= 0.:
        raise ValueError('exit point is undefined')

    # figure out when it would reach each face
    if vx != 0.:
        xt = abs(distx/vx)
    else:
        xt = math.inf
    if vy != 0.:
        yt = abs(disty/vy)
    else:
        yt = math.inf
    if vz != 0.:
        zt = abs(distz/vz)
    else:
        zt = math.inf

    # find which face it reaches first
    mint = min([xt, yt, zt])

    # calculate the position when it reaches that face
    xout = xin + vx*mint
    yout = yin + vy*mint
    zout = zin + vz*mint

    # make sure exit point is actually on the box
    if xmin <= xout <= xmax and ymin <= yout <= ymax and zmin <= zout <= zmax:
        return xout, yout, zout
    else:
        print('exit point not part of box')
        return None, None, None


if __name__ == '__main__':
    print(line_box_clipper3d(0., 0., 0., 1., 1., 1., 0.2, 0.5, 0.3, 3, 0.5, 1))
