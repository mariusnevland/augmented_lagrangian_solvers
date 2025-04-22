import porepy as pp
Scalar = pp.ad.Scalar


class NormalPermeabilityFromSecondary:
    """Introduce the cubic law for the normal permeability."""

    def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        aperture = self.aperture(subdomains)
        permeability = (aperture ** Scalar(2)) / Scalar(12)
        normal_perm = projection.secondary_to_mortar_avg() @ permeability
        return normal_perm