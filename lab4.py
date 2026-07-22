import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pp


def ClassicRK(t, f, y0):
    y = np.zeros(np.size(t))
    y[0] = y0

    for k in range(np.size(t) - 1):
        h = t[k + 1] - t[k]
        k1 = h * f(t[k], y[k])
        k2 = h * f(t[k] + h / 2, y[k] + k1 / 2)
        k3 = h * f(t[k] + h / 2, y[k] + k2 / 2)
        k4 = h * f(t[k] + h, y[k] + k3)
        y[k + 1] = y[k] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y


def funk(t, y):
    return -y


def Effekt(t):
    if t <= 1.0:
        return 10000
    return 0


def GlodTrad(t, T):
    T_inf = 18 + 273.15
    return Effekt(t) - 2.76e-10 * (T**4 - T_inf**4)


def uppgift_2_2_och_2_3():
    print("Uppgift 2.2")
    print(" n        h          y(1;h)")

    y_slut = []
    for n in [5, 10, 20]:
        t = np.linspace(0, 1, n + 1)
        y = ClassicRK(t, funk, 1)
        y_slut.append(y[-1])
        print("{:2d}   {:8.5f}   {:.10f}".format(n, 1 / n, y[-1]))

    kvot = (y_slut[0] - y_slut[1]) / (y_slut[1] - y_slut[2])
    p = np.log(abs(kvot)) / np.log(2)

    print()
    print("Uppgift 2.3")
    print("Kvot = {:.6f}".format(kvot))
    print("p = {:.6f}".format(p))
    print("Teorin säger p = 4 för klassiska Runge-Kutta-metoden.")
    print()


def uppgift_2_4_och_2_5():
    n = 100
    t = np.linspace(0, 5, n + 1)
    e = np.zeros(np.size(t))

    for k in range(np.size(t)):
        e[k] = Effekt(t[k])

    fig = pp.figure(figsize=(8, 5), dpi=150)
    pp.plot(t, e)
    pp.xlabel("t [s]")
    pp.ylabel("E(t)")
    pp.title("Tillförd effekt")
    pp.grid(True)
    fig.tight_layout()
    pp.savefig("uppgift_2_4_effekt.png")
    pp.close()

    T0 = 18 + 273.15
    T = ClassicRK(t, GlodTrad, T0)
    T_celsius = T - 273.15

    fig = pp.figure(figsize=(8, 5), dpi=150)
    pp.plot(t, T_celsius)
    pp.xlabel("t [s]")
    pp.ylabel("T [°C]")
    pp.title("Glödtrådens temperatur")
    pp.grid(True)
    fig.tight_layout()
    pp.savefig("uppgift_2_5_glodtrad.png")
    pp.close()

    print("Uppgift 2.4 och 2.5")
    print("Grafer sparade i uppgift_2_4_effekt.png och uppgift_2_5_glodtrad.png")
    print("T(0) = {:.2f} C".format(T_celsius[0]))
    print("T(1) = {:.2f} C".format(T_celsius[20]))
    print("T(5) = {:.2f} C".format(T_celsius[-1]))
    print("Uppgift 2.6: Modellen visar snabb uppvarmning och langsammare avsvalning.")
    print()


def main():
    uppgift_2_2_och_2_3()
    uppgift_2_4_och_2_5()


if __name__ == "__main__":
    main()
