import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pp
from mpl_toolkits import mplot3d
from scipy.integrate import odeint


m = 0.3056
g = 9.81
rho = 1.2
A_area = 0.0375
Cd = 0.0
Cl = 0.0
omega = np.array([0.0, 0.0, 0.0])


def BollODE(S, t):
    P = S[0:3]
    V = S[3:6]

    G = np.array([0, 0, -g])
    Fm = m * G

    fart = np.sqrt(V[0] ** 2 + V[1] ** 2 + V[2] ** 2)
    Fd = -0.5 * rho * A_area * Cd * fart * V

    Fl = Cl * rho * A_area * np.cross(omega, V)

    F = Fm + Fd + Fl
    acceleration = F / m

    dS = np.zeros(6)
    dS[0:3] = V
    dS[3:6] = acceleration
    return dS


def HittaRelevantaVarden(z):
    for k in range(np.size(z)):
        if z[k] < 0:
            return k
    return np.size(z) - 1


def landningspunkt(BollBanan):
    z = BollBanan[:, 2]

    for k in range(1, np.size(z)):
        if z[k] < 0:
            andel = z[k - 1] / (z[k - 1] - z[k])
            punkt = BollBanan[k - 1, 0:3] + andel * (
                BollBanan[k, 0:3] - BollBanan[k - 1, 0:3]
            )
            return punkt

    return BollBanan[-1, 0:3]


def simulera(P0, V0, cd, cl, omega_varde, n=500, t_slut=5):
    global Cd, Cl, omega

    Cd = cd
    Cl = cl
    omega = omega_varde

    S0 = np.zeros(6)
    S0[0:3] = P0
    S0[3:6] = V0

    t = np.linspace(0, t_slut, n)
    BollBanan = odeint(BollODE, S0, t)
    return t, BollBanan


def plotta_bana(BollBanan, filnamn, titel, visa_mal=False):
    k = HittaRelevantaVarden(BollBanan[:, 2])

    if visa_mal:
        fig, ax = pp.subplots(2, 1, figsize=(8, 7), dpi=150)

        ax[0].plot(BollBanan[0 : k + 1, 0], BollBanan[0 : k + 1, 2], "blue")
        ax[0].plot([35.3, 35.3], [0, 2.44], "red")
        ax[0].set_xlabel("x [m]")
        ax[0].set_ylabel("z [m]")
        ax[0].set_title(titel + " - sidovy")
        ax[0].grid(True)

        ax[1].plot(BollBanan[0 : k + 1, 0], BollBanan[0 : k + 1, 1], "blue")
        ax[1].plot([35.3, 35.3], [-7.32 / 2, 7.32 / 2], "red")
        ax[1].set_xlabel("x [m]")
        ax[1].set_ylabel("y [m]")
        ax[1].set_title(titel + " - ovanvy")
        ax[1].grid(True)
    else:
        fig, ax = pp.subplots(figsize=(8, 5), dpi=150)
        ax.plot(BollBanan[0 : k + 1, 0], BollBanan[0 : k + 1, 2], "blue")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(titel)
        ax.grid(True)

    fig.tight_layout()
    pp.savefig(filnamn)
    pp.close()


def plotta_bana_3d_med_mal(BollBanan, filnamn, titel):
    fig = pp.figure(figsize=(9, 6), dpi=150)
    ax = pp.axes(projection="3d")
    k = HittaRelevantaVarden(BollBanan[:, 2])

    ax.plot3D(BollBanan[0 : k + 1, 0], BollBanan[0 : k + 1, 1], BollBanan[0 : k + 1, 2], "blue")
    ax.plot3D(
        35.3 * np.array([1, 1, 1, 1]),
        7.32 * np.array([-1, -1, 1, 1]) / 2,
        2.44 * np.array([0, 1, 1, 0]),
        "red",
    )

    ax.set_xlabel("x [m]", labelpad=10)
    ax.set_ylabel("y [m]", labelpad=10)
    ax.set_zlabel("z [m]", labelpad=10)
    ax.set_title(titel)
    ax.view_init(elev=24, azim=-62)
    ax.set_box_aspect([3, 1, 1])
    fig.tight_layout()
    pp.savefig(filnamn)
    pp.close()


def y_och_z_vid_x(BollBanan, x_mal):
    for k in range(1, np.shape(BollBanan)[0]):
        if BollBanan[k, 0] >= x_mal:
            andel = (x_mal - BollBanan[k - 1, 0]) / (BollBanan[k, 0] - BollBanan[k - 1, 0])
            punkt = BollBanan[k - 1, 0:3] + andel * (
                BollBanan[k, 0:3] - BollBanan[k - 1, 0:3]
            )
            return punkt[1], punkt[2]

    return BollBanan[-1, 1], BollBanan[-1, 2]


def bestam_cd():
    P0 = np.array([0, 0, 0])
    V0 = np.array([17.32, 0.45, 11.51])
    target_x = 23.50

    a = 0.0
    b = 1.0

    for k in range(40):
        c = (a + b) / 2
        t, bana = simulera(P0, V0, c, 0.0, np.array([0.0, 0.0, 0.0]), 1000, 5)
        landning = landningspunkt(bana)

        if landning[0] > target_x:
            a = c
        else:
            b = c

    return (a + b) / 2


def uppgift_3_2():
    vx0 = 17.3
    vz0 = 11.5
    landningstid = 2 * vz0 / g
    landnings_x = vx0 * landningstid

    print("Uppgift 3.2")
    print("Analytiskt landar bollen vid x = {:.3f} m, y = 0 m".format(landnings_x))
    print()


def uppgift_3_1():
    print("Uppgift 3.1")
    print("dS/dt = [vx, vy, vz, Fx/m, Fy/m, Fz/m]")
    print()


def uppgift_3_3():
    P0 = np.array([0, 0, 0])
    V0 = np.array([17.3, 0, 11.5])
    t, bana = simulera(P0, V0, 0.0, 0.0, np.array([0.0, 0.0, 0.0]))
    landning = landningspunkt(bana)
    plotta_bana(bana, "uppgift_3_3_gravitation.png", "Bollbana med gravitation")

    print("Uppgift 3.3")
    print("Simuleringen landar vid x = {:.3f} m, y = {:.3f} m".format(landning[0], landning[1]))
    print()


def uppgift_3_4():
    P0 = np.array([0, 0, 0])
    V0 = np.array([17.3, 0, 11.5])
    t, bana = simulera(P0, V0, 0.17, 0.0, np.array([0.0, 0.0, 0.0]))
    landning = landningspunkt(bana)
    plotta_bana(bana, "uppgift_3_4_luftmotstand.png", "Bollbana med luftmotstand")

    print("Uppgift 3.4")
    print("Med luftmotstand landar bollen vid x = {:.3f} m, y = {:.3f} m".format(landning[0], landning[1]))
    print()


def uppgift_3_5():
    cd = bestam_cd()
    P0 = np.array([0, 0, 0])
    V0 = np.array([17.32, 0.45, 11.51])
    t, bana = simulera(P0, V0, cd, 0.0, np.array([0.0, 0.0, 0.0]), 1000, 5)
    landning = landningspunkt(bana)

    print("Uppgift 3.5")
    print("CD = {:.4f}".format(cd))
    print("Kontroll: landning vid x = {:.3f} m, y = {:.3f} m".format(landning[0], landning[1]))
    print()


def uppgift_3_6_och_3_7():
    P0 = np.array([0, 0, 0])
    V0 = np.array([35.4, -14.9, 4.13])
    omega_varde = np.array([0.0, -8.9, 23.6])
    t, bana = simulera(P0, V0, 0.14, 0.16, omega_varde, 1500, 5)
    plotta_bana(bana, "uppgift_3_6_skruv.png", "Bollbana med Magnuskraft", True)
    plotta_bana_3d_med_mal(bana, "uppgift_3_6_skruv_3d.png", "Bollbana med Magnuskraft")
    y_mal, z_mal = y_och_z_vid_x(bana, 35.3)
    rotationshastighet = np.sqrt(np.sum(omega_varde**2)) / (2 * np.pi)

    print("Uppgift 3.6 och 3.7")
    print("Vald omega = [{:.2f}, {:.2f}, {:.2f}]".format(omega_varde[0], omega_varde[1], omega_varde[2]))
    print("Vid malet: y = {:.3f} m, z = {:.3f} m".format(y_mal, z_mal))
    print("Rotationshastighet = {:.3f} varv/s".format(rotationshastighet))
    print()

    return omega_varde


def uppgift_3_8(omega_varde):
    P0 = np.array([0, 0, 0])
    V0 = np.array([35.4, -14.9, 4.13])

    t, bana_015 = simulera(P0, V0, 0.14, 0.15, omega_varde, 1500, 5)
    t, bana_017 = simulera(P0, V0, 0.14, 0.17, omega_varde, 1500, 5)

    landning_015 = landningspunkt(bana_015)
    landning_017 = landningspunkt(bana_017)
    skillnad = landning_017 - landning_015
    avstand = np.sqrt(skillnad[0] ** 2 + skillnad[1] ** 2)

    print("Uppgift 3.8")
    print("CL = 0.15: landning x = {:.3f}, y = {:.3f}".format(landning_015[0], landning_015[1]))
    print("CL = 0.17: landning x = {:.3f}, y = {:.3f}".format(landning_017[0], landning_017[1]))
    print("Nedslagsplatsen flyttas {:.3f} m i xy-planet.".format(avstand))
    print()


def main():
    uppgift_3_1()
    uppgift_3_2()
    uppgift_3_3()
    uppgift_3_4()
    uppgift_3_5()
    omega_varde = uppgift_3_6_och_3_7()
    uppgift_3_8(omega_varde)
    print("Uppgift 3.9")
    print("En enkel forbattrning ar att lata Cd och Cl bero pa bollens hastighet och rotation.")


if __name__ == "__main__":
    main()
