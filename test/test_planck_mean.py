from meshoid.radiation.dust import dust_mean_opacity
import numpy as np
from matplotlib import pyplot as plt

Trad = np.logspace(1, 4, 10**4)
Tdust_vals = 10, 200, 300, 500, 1000
colors = plt.get_cmap("inferno")(np.linspace(0, 0.8, len(Tdust_vals)))

for i, Td in enumerate(Tdust_vals):
    kappa = dust_mean_opacity(Trad, Td)
    fit = np.polyfit(np.log10(Trad[Trad < 1e3]), np.log10(kappa[Trad < 1e3]), 6)
    plt.loglog(Trad, kappa, label=r"$T_{\rm dust}=%g\rm K$" % Td, color=colors[i])
    kappa_fit = 10 ** np.polyval(fit, np.log10(Trad))
    plt.loglog(Trad[::2000], kappa[::2000], color=colors[i], ls="dashed")

kappa_LTE = dust_mean_opacity(Trad, Trad)
kappa_LTE[kappa_LTE == 0] = np.nan
plt.loglog(Trad, kappa_LTE, label=r"$T_{\rm rad}=T_{\rm dust}$", color="black", ls="dashed")
plt.ylabel(r"$\kappa_{\rm P} \,\left(\rm cm^{2}\,g^{-1}\right)$")
plt.xlabel(r"$T_{\rm rad} \,\left(\rm K\right)$")
plt.legend()
plt.savefig("Trad_vs_kappa_planck.pdf", bbox_inches="tight")
plt.clf()

Trad = np.logspace(1, 4, 10**4)
for i, Td in enumerate(Tdust_vals):
    kappa = dust_mean_opacity(Trad, Td, which="rosseland")
    plt.loglog(Trad, kappa, label=r"$T_{\rm dust}=%g\rm K$" % Td, color=colors[i])

kappa_LTE = dust_mean_opacity(Trad, Trad, which="rosseland")
kappa_LTE[kappa_LTE == 0] = np.nan
plt.loglog(Trad, kappa_LTE, label=r"$T_{\rm rad}=T_{\rm dust}$", color="black", ls="dashed")
plt.ylabel(r"$\kappa_{\rm R} \,\left(\rm cm^{2}\,g^{-1}\right)$")
plt.xlabel(r"$T_{\rm rad} \,\left(\rm K\right)$")
plt.legend()
plt.savefig("Trad_vs_kappa_rosseland.pdf", bbox_inches="tight")
plt.clf()
