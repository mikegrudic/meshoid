from meshoid.radiation.dust import dust_mean_opacity
import numpy as np
from matplotlib import pyplot as plt

# Trad = np.logspace(0, 4, 10**4)
# Tdust_vals = 10, 200, 300, 500, 1000
# Tdust_intervals = ((0, 160), (160, 275), (275, 425), (425, 680), (680, 1500))
# colors = plt.get_cmap("inferno")(np.linspace(0.2, 0.8, len(Tdust_vals)))

# for i, (Td1, Td2) in enumerate(Tdust_intervals):
#     kappa = dust_mean_opacity(Trad, Td1)
#     fit = np.polyfit(np.log10(Trad), np.log10(kappa), 9)
#     plt.loglog(Trad, kappa, label=r"$T_{\rm d }\in (%g,%g) \rm K$" % (Td1, Td2), color=colors[i])

#     plt.loglog(Trad, 10.0 ** np.polyval(fit, np.log10(Trad)), ls="dotted", color=colors[i])

# kappa_LTE = dust_mean_opacity(Trad, Trad)
# # kappa_LTE[kappa_LTE == 0] = np.nan
# plt.xlim(2.73, 1e4)
# # plt.ylim(1e-2, 100)
# plt.loglog(Trad, kappa_LTE, label=r"$T_{\rm rad}=T_{\rm dust}$", color="black", ls="dashed")
# plt.ylabel(r"$\kappa_{\rm P} \,\left(\rm cm^{2}\,g^{-1}\right)$")
# plt.xlabel(r"$T_{\rm rad} \,\left(\rm K\right)$")
# legend_args = {"frameon": True, "labelspacing": 0}
# plt.legend(**legend_args)
# plt.savefig("Trad_vs_kappa_planck.pdf", bbox_inches="tight")
# plt.clf()

# Trad = np.logspace(1, 4, 10**4)
# for i, (Td1, Td2) in enumerate(Tdust_intervals):
#     kappa = dust_mean_opacity(Trad, Td1, which="rosseland")
#     plt.loglog(Trad, kappa, label=r"$T_{\rm d }\in (%g,%g) \rm K$" % (Td1, Td2), color=colors[i])

# kappa_LTE = dust_mean_opacity(Trad, Trad, which="rosseland")
# # kappa_LTE[kappa_LTE == 0] = np.nan
# plt.xlim(10, 1e4)
# plt.ylim(1e-3, 100)
# plt.loglog(Trad, kappa_LTE, label=r"$T_{\rm rad}=T_{\rm dust}$", color="black", ls="dashed")
# plt.ylabel(r"$\kappa_{\rm R} \,\left(\rm cm^{2}\,g^{-1}\right)$")
# plt.xlabel(r"$T_{\rm rad} \,\left(\rm K\right)$")
# plt.legend(**legend_args)
# plt.savefig("Trad_vs_kappa_rosseland.pdf", bbox_inches="tight")
# plt.clf()
