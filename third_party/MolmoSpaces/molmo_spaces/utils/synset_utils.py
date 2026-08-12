from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from functools import cache, lru_cache


def _ensure_nltk():
    import nltk

    for corpus in ["wordnet", "wordnet2022"]:
        nltk.download(corpus)


_ensure_nltk()

from nltk.corpus import wordnet2022 as wn

wn.abspaths()

from nltk.corpus.reader import Synset

from molmo_spaces.utils.constants.object_constants import AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET
from molmo_spaces.utils.object_metadata import ObjectMeta

# new excluded hypernyms from
# https://github.com/allenai/holodeck-internal/blob/2b3da34c8592525b93ec566d6a10bdefd72baf2a/ai2holodeck/asset_filtering/synset_utils.py#L102
# Updated to remove concrete objects and add new abstract synsets for better lemma filtering
EXCLUDED_HYPERNYMS = frozenset(
    {
        "abstraction.n.04",
        "abstraction.n.06",
        "accident.n.01",
        "accomplice.n.01",
        "accumulation.n.04",
        "act.n.02",
        "acting.n.01",
        "action.n.01",
        "action.n.07",
        "activity.n.01",
        "administrative_unit.n.01",
        "admirer.n.03",
        "adult.n.01",
        "affair.n.03",
        "agaric.n.02",
        "agglomeration.n.01",
        "air_unit.n.01",
        "alloy.n.01",
        "animal_material.n.01",
        "animal_order.n.01",
        "animal_product.n.01",
        "announcement.n.02",
        "anomaly.n.02",
        "aperture.n.03",
        "appearance.n.01",
        "appearance.n.02",
        "appearance.n.04",
        "application.n.03",
        "approval.n.04",
        "archosaur.n.01",
        "arctiid.n.01",
        "area.n.05",
        "area.n.06",
        "aristocrat.n.01",
        "army_unit.n.01",
        "arrangement.n.02",
        "arrangement.n.03",
        "art.n.03",
        "art_form.n.01",
        "arthropod_family.n.01",
        "arthropod_genus.n.01",
        "article.n.02",
        "articulator.n.02",
        "artifact.n.01",
        "artificial_intelligence.n.01",
        "artificial_language.n.01",
        "artillery.n.02",
        "artistic_style.n.01",
        "asphodel.n.01",
        "assembly.n.01",
        "assembly.n.05",
        "assembly.n.06",
        "assets.n.01",
        "assistant.n.01",
        "associate.n.01",
        "association.n.08",
        "atom.n.02",
        "atomic_theory.n.01",
        "attempt.n.01",
        "attendant.n.01",
        "attitude.n.01",
        "attribute.n.02",
        "auditory_communication.n.01",
        "autoloader.n.01",
        "automatic_firearm.n.01",
        "avoirdupois_unit.n.01",
        "axis.n.06",
        "back.n.08",
        "base.n.01",
        "basic_cognitive_process.n.01",
        "basidiomycete.n.01",
        "beginning.n.05",
        "being.n.01",
        "belief.n.01",
        "benzene.n.01",
        "bill.n.07",
        "binary_compound.n.01",
        "bioassay.n.01",
        "biological_group.n.01",
        "biometric_identification.n.01",
        "blemish.n.01",
        "body.n.02",
        "body.n.04",
        "body_part.n.01",
        "bodybuilding.n.01",
        "boundary.n.01",
        "bowling.n.01",
        "bramble_bush.n.01",
        "bryophyte.n.01",
        "business.n.01",
        "businessperson.n.01",
        "calcium_carbonate.n.01",
        "calcium_sulphate.n.01",
        "capitalist.n.02",
        "capsule.n.03",
        "capsule.n.05",
        "carbon.n.01",
        "care.n.01",
        "caryophylloid_dicot_genus.n.01",
        "category.n.02",
        "catholic_church.n.01",
        "causal_agent.n.01",
        "cavity.n.02",
        "center.n.01",
        "center.n.04",
        "center.n.06",
        "central.n.01",
        "ceratopsian.n.01",
        "cetacean.n.01",
        "change.n.03",
        "change_of_location.n.01",
        "change_of_state.n.01",
        "character.n.04",
        "character.n.08",
        "chemical_phenomenon.n.01",
        "chemoreceptor.n.01",
        "chicory.n.04",
        "child.n.01",
        "child.n.02",
        "chordate.n.01",
        "circle.n.01",
        "class.n.03",
        "clef.n.01",
        "clown.n.02",
        "clue.n.02",
        "code.n.03",
        "coding_system.n.01",
        "cognition.n.01",
        "cognitive_factor.n.01",
        "collection.n.01",
        "collision.n.02",
        "color.n.01",
        "combatant.n.01",
        "comedian.n.01",
        "commodity.n.01",
        "communication.n.02",
        "complexity.n.01",
        "component.n.03",
        "composition.n.03",
        "compound_leaf.n.01",
        "compression.n.04",
        "computer_graphics.n.01",
        "computer_network.n.01",
        "computer_science.n.01",
        "concealment.n.03",
        "concept.n.01",
        "conduit.n.01",
        "confinement.n.03",
        "conic_section.n.01",
        "connection.n.01",
        "consequence.n.01",
        "constitution.n.04",
        "constraint.n.01",
        "consumer_credit.n.01",
        "consumer_goods.n.01",
        "content.n.05",
        "contestant.n.01",
        "control.n.05",
        "convex_shape.n.01",
        "cook.n.01",
        "cooking.n.01",
        "cookout.n.01",
        "coordinate_system.n.01",
        "copper-base_alloy.n.01",
        "correctional_institution.n.01",
        "corrective.n.01",
        "course.n.08",
        "covering.n.02",
        "crack.n.07",
        "craftsman.n.03",
        "creating_by_removal.n.01",
        "creating_from_raw_materials.n.01",
        "creation.n.01",
        "creation.n.02",
        "creator.n.02",
        "crest.n.05",
        "criminal.n.01",
        "cross_section.n.01",
        "crossing.n.05",
        "crossopterygian.n.01",
        "crosspiece.n.02",
        "cuisine.n.01",
        "cultivation.n.02",
        "cyprinodont.n.01",
        "danaid.n.01",
        "dance_music.n.02",
        "dark.n.01",
        "database.n.01",
        "decapod.n.02",
        "deceiver.n.01",
        "decline.n.02",
        "decorativeness.n.01",
        "defender.n.01",
        "definite_quantity.n.01",
        "deity.n.01",
        "delicious.n.01",
        "delivery.n.01",
        "demonstration.n.05",
        "depiction.n.04",
        "depository.n.01",
        "depression.n.08",
        "design.n.02",
        "design.n.04",
        "detail.n.02",
        "determinant.n.01",
        "development.n.06",
        "device.n.01",
        "diapsid.n.01",
        "dicot_genus.n.01",
        "diet.n.01",
        "difficulty.n.02",
        "digit.n.01",
        "direction.n.06",
        "discharge.n.03",
        "discipline.n.01",
        "discrimination.n.02",
        "disorderliness.n.01",
        "display.n.05",
        "district.n.01",
        "ditch.n.01",
        "diver.n.01",
        "division.n.03",
        "division.n.04",
        "dresser.n.02",
        "drive.n.02",
        "drop.n.01",
        "dry_masonry.n.01",
        "dryad.n.01",
        "durables.n.01",
        "dwelling.n.01",
        "dysphemism.n.01",
        "ectoparasite.n.01",
        "edge.n.03",
        "edge.n.06",
        "edging.n.01",
        "effect.n.03",
        "effort.n.02",
        "egotist.n.01",
        "elasmobranch.n.01",
        "elasticity.n.01",
        "electronic_text.n.01",
        "elite.n.01",
        "ellipse.n.01",
        "embankment.n.01",
        "emoticon.n.01",
        "employee.n.01",
        "enamel.n.04",
        "enclosure.n.03",
        "engineering.n.02",
        "enlisted_person.n.01",
        "enterprise.n.02",
        "entertainment.n.01",
        "entity.n.01",
        "entree.n.01",
        "escape.n.05",
        "eubacteria.n.01",
        "european.n.01",
        "evaluator.n.01",
        "even-toed_ungulate.n.01",
        "event.n.01",
        "evil_spirit.n.01",
        "example.n.01",
        "excretory_organ.n.01",
        "exercise.n.01",
        "exhibitionist.n.02",
        "expanse.n.03",
        "expedient.n.01",
        "experience.n.02",
        "explanation.n.02",
        "explorer.n.01",
        "external_body_part.n.01",
        "extremity.n.01",
        "extremity.n.05",
        "extremum.n.02",
        "exudate.n.01",
        "facial_expression.n.01",
        "facial_hair.n.01",
        "facility.n.04",
        "facing.n.03",
        "failure.n.02",
        "family.n.06",
        "fancier.n.01",
        "fare.n.04",
        "farming.n.01",
        "fashion.n.03",
        "feature.n.02",
        "feline.n.01",
        "female.n.02",
        "fern_ally.n.01",
        "ferric_oxide.n.01",
        "fibril.n.01",
        "fiction.n.01",
        "field.n.01",
        "figuration.n.02",
        "financial_gain.n.01",
        "fine_arts.n.01",
        "finish.n.04",
        "fire.n.01",
        "firing_range.n.01",
        "first_class.n.02",
        "flow.n.01",
        "flue.n.03",
        "fluid.n.02",
        "font.n.01",
        "foothold.n.02",
        "force.n.02",
        "forest.n.01",
        "formation.n.01",
        "formula.n.04",
        "formulation.n.01",
        "foundry.n.01",
        "framework.n.03",
        "front.n.04",
        "fruitwood.n.01",
        "fullerene.n.01",
        "fundamental_quantity.n.01",
        "gadoid.n.01",
        "gain.n.04",
        "game_of_chance.n.01",
        "gang.n.03",
        "ganoid.n.01",
        "gas.n.02",
        "gastropod.n.01",
        "gate.n.04",
        "genre.n.03",
        "genus.n.02",
        "geographic_point.n.01",
        "geographical_area.n.01",
        "geometry.n.01",
        "girdle.n.01",
        "glyptic_art.n.01",
        "golf.n.01",
        "goosefoot.n.01",
        "graphics.n.02",
        "greco-roman_deity.n.01",
        "greek_deity.n.01",
        "grip.n.06",
        "groove.n.01",
        "group.n.01",
        "group_action.n.01",
        "hair.n.01",
        "happening.n.01",
        "hawkmoth.n.01",
        "hazard.n.01",
        "head.n.04",
        "health_hazard.n.01",
        "health_professional.n.01",
        "heating.n.01",
        "hexagram.n.01",
        "higher_cognitive_process.n.01",
        "hiker.n.01",
        "hill.n.01",
        "hindrance.n.01",
        "hindrance.n.02",
        "hindu_deity.n.01",
        "history.n.02",
        "hole.n.01",
        "hole.n.02",
        "hole.n.05",
        "homespun.n.01",
        "homo.n.02",
        "horn.n.07",
        "housing.n.01",
        "humate.n.01",
        "humorist.n.01",
        "hunting_dog.n.01",
        "hydrocarbon.n.01",
        "hydrozoan.n.01",
        "hypothesis.n.02",
        "idea.n.01",
        "ideal.n.01",
        "idler.n.01",
        "illumination.n.02",
        "illusion.n.01",
        "illustration.n.01",
        "imaginary_place.n.01",
        "imagination.n.02",
        "imaging.n.02",
        "immateriality.n.02",
        "implement.n.01",
        "implementation.n.02",
        "impression.n.01",
        "incident.n.01",
        "income.n.01",
        "indefinite_quantity.n.01",
        "individual.n.02",
        "industry.n.02",
        "influence.n.01",
        "information.n.02",
        "inhabitant.n.01",
        "insertion.n.02",
        "institution.n.01",
        "intake.n.02",
        "integer.n.01",
        "intellectual.n.01",
        "interior_decoration.n.02",
        "inventiveness.n.01",
        "investigator.n.02",
        "iron.n.01",
        "isogon.n.01",
        "isopod.n.01",
        "item.n.01",
        "item.n.02",
        "item.n.03",
        "item.n.04",
        "item.n.05",
        "item.n.06",
        "jack.n.11",
        "jail.n.01",
        "junction.n.04",
        "juvenile.n.01",
        "juxtaposition.n.01",
        "killer.n.01",
        "kind.n.01",
        "kingdom.n.01",
        "knowledge_domain.n.01",
        "labor.n.02",
        "laborer.n.01",
        "lake.n.01",
        "lamination.n.01",
        "land.n.01",
        "landing.n.02",
        "lane.n.02",
        "language.n.01",
        "language_unit.n.01",
        "larid.n.01",
        "latex.n.01",
        "lawman.n.01",
        "layer.n.02",
        "leader.n.01",
        "leg.n.02",
        "legend.n.01",
        "leporid.n.01",
        "level.n.05",
        "life_science.n.01",
        "lignite.n.01",
        "likeness.n.02",
        "liliid_monocot_genus.n.01",
        "limit.n.04",
        "limit.n.06",
        "linear_unit.n.01",
        "lipid.n.01",
        "liquid.n.03",
        "list.n.01",
        "literary_composition.n.01",
        "literate.n.01",
        "living_quarters.n.01",
        "living_thing.n.01",
        "local_area_network.n.01",
        "location.n.01",
        "lookout.n.02",
        "lottery.n.02",
        "lover.n.01",
        "machine.n.02",
        "macromolecule.n.01",
        "magnitude.n.01",
        "main.n.02",
        "male_aristocrat.n.01",
        "male_child.n.01",
        "malformation.n.02",
        "man.n.01",
        "manner.n.01",
        "manual_labor.n.01",
        "mark.n.04",
        "marking.n.02",
        "martial_art.n.01",
        "mass_unit.n.01",
        "material.n.01",
        "material.n.04",
        "mathematics.n.01",
        "matter.n.01",
        "matter.n.02",
        "matter.n.03",
        "matter.n.06",
        "means.n.01",
        "measure.n.02",
        "mechanism.n.05",
        "medical_procedure.n.01",
        "meeting.n.01",
        "membrane.n.02",
        "merchant.n.01",
        "metallic_element.n.01",
        "message.n.01",
        "message.n.02",
        "military_quarters.n.01",
        "military_unit.n.01",
        "minimum.n.01",
        "misconception.n.01",
        "misfortune.n.01",
        "mishap.n.02",
        "mixture.n.01",
        "molecular_formula.n.01",
        "moneran.n.01",
        "monetary_unit.n.01",
        "monocot_genus.n.01",
        "motion.n.06",
        "motor_hotel.n.01",
        "movement.n.03",
        "movement.n.04",
        "movement.n.11",
        "multidimensional_language.n.01",
        "murderer.n.01",
        "music.n.01",
        "musical_composition.n.01",
        "musical_notation.n.01",
        "musical_organization.n.01",
        "musician.n.01",
        "muslim.n.01",
        "name.n.01",
        "natural_elevation.n.01",
        "natural_object.n.01",
        "natural_phenomenon.n.01",
        "natural_process.n.01",
        "natural_science.n.01",
        "negotiator.n.01",
        "neritid.n.01",
        "net_income.n.01",
        "nidus.n.02",
        "nobility.n.01",
        "noise.n.01",
        "nongovernmental_organization.n.01",
        "nonmetal.n.01",
        "nonworker.n.01",
        "notation.n.01",
        "notion.n.04",
        "number.n.02",
        "nutrient.n.02",
        "object.n.01",
        "object.n.04",
        "obstacle.n.01",
        "occultist.n.01",
        "occupation.n.01",
        "offspring.n.01",
        "oil_paint.n.01",
        "oldster.n.01",
        "open_chain.n.01",
        "operation.n.06",
        "orchis.n.01",
        "order.n.12",
        "order.n.14",
        "organelle.n.01",
        "organic_compound.n.01",
        "organism.n.01",
        "organization.n.01",
        "orifice.n.01",
        "originality.n.01",
        "ornithischian.n.01",
        "orthography.n.01",
        "oscine.n.01",
        "ovule.n.01",
        "oxide.n.01",
        "pad.n.02",
        "padding.n.01",
        "parallelepiped.n.01",
        "parasite.n.01",
        "paring.n.01",
        "part.n.01",
        "part.n.02",
        "part.n.03",
        "partial_veil.n.01",
        "participant.n.01",
        "particle.n.02",
        "particulate.n.01",
        "passage.n.03",
        "patron_saint.n.01",
        "pedaler.n.01",
        "peer.n.01",
        "percept.n.01",
        "perception.n.03",
        "percussionist.n.01",
        "performance.n.02",
        "performer.n.01",
        "performing_arts.n.01",
        "perpendicular.n.02",
        "personal_property.n.01",
        "phenomenon.n.01",
        "physical_entity.n.01",
        "physical_phenomenon.n.01",
        "physical_property.n.01",
        "pictorial_representation.n.01",
        "piece.n.01",
        "placement.n.01",
        "plain.n.01",
        "plan.n.01",
        "plan.n.03",
        "plane_figure.n.01",
        "plant_genus.n.01",
        "plant_order.n.01",
        "plant_organ.n.01",
        "plant_part.n.01",
        "plant_process.n.01",
        "play.n.08",
        "player.n.01",
        "point_of_view.n.01",
        "porcelain.n.01",
        "portrayal.n.02",
        "poseur.n.01",
        "position.n.07",
        "position.n.12",
        "post.n.01",
        "power.n.01",
        "practice.n.01",
        "practice_range.n.01",
        "prayer.n.02",
        "presence.n.01",
        "presentation.n.02",
        "preserver.n.03",
        "principal.n.05",
        "problem.n.02",
        "procedure.n.01",
        "process.n.02",
        "process.n.05",
        "process.n.06",
        "prod.n.02",
        "product.n.02",
        "production.n.02",
        "production.n.07",
        "profile.n.05",
        "program.n.07",
        "programming_language.n.01",
        "projection.n.04",
        "property.n.01",
        "property.n.02",
        "property.n.04",
        "property.n.05",
        "proportional_font.n.01",
        "propulsion.n.01",
        "propulsion.n.02",
        "protection.n.01",
        "protocol.n.01",
        "protoctist.n.01",
        "psychological_feature.n.01",
        "public_square.n.01",
        "punctuation.n.02",
        "pure_mathematics.n.01",
        "push.n.01",
        "quality.n.01",
        "railway.n.01",
        "range.n.04",
        "range.n.05",
        "ration.n.01",
        "reaction_propulsion.n.01",
        "real_property.n.01",
        "rectangle.n.01",
        "region.n.01",
        "region.n.03",
        "regular_polygon.n.01",
        "relation.n.01",
        "relationship.n.03",
        "relative.n.01",
        "religion.n.02",
        "repair_shop.n.01",
        "representation.n.01",
        "representation.n.02",
        "representational_process.n.01",
        "representative.n.01",
        "reproductive_cell.n.01",
        "reproductive_structure.n.01",
        "reptile_family.n.01",
        "reptile_genus.n.01",
        "reserve.n.02",
        "residential_district.n.01",
        "residue.n.01",
        "resin.n.01",
        "resource.n.03",
        "respiratory_tract.n.01",
        "restoration.n.06",
        "retreat.n.02",
        "rider.n.03",
        "rig.n.03",
        "right.n.01",
        "robotics.n.01",
        "roman_deity.n.01",
        "room.n.02",
        "rosid_dicot_genus.n.01",
        "rotating_mechanism.n.01",
        "row.n.01",
        "rubber.n.01",
        "ruminant.n.01",
        "saint.n.01",
        "salmonid.n.01",
        "salt.n.01",
        "sample.n.03",
        "sanitary_condition.n.01",
        "satirist.n.01",
        "saurischian.n.01",
        "saying.n.01",
        "scholar.n.01",
        "school.n.04",
        "science.n.01",
        "scientific_theory.n.01",
        "scorpaenid.n.01",
        "script.n.03",
        "section.n.03",
        "section.n.04",
        "section.n.08",
        "sediment.n.01",
        "self-defense.n.01",
        "semipermeable_membrane.n.01",
        "sense_organ.n.01",
        "serviceman.n.01",
        "set.n.13",
        "setting.n.02",
        "settlement.n.06",
        "sewing.n.02",
        "shaft.n.08",
        "shape.n.02",
        "sheath.n.02",
        "sheet.n.06",
        "shell.n.02",
        "show.n.01",
        "side.n.04",
        "side.n.05",
        "side.n.09",
        "sign.n.01",
        "sign.n.11",
        "signal.n.01",
        "silhouette.n.02",
        "situation.n.01",
        "skating.n.01",
        "skilled_worker.n.01",
        "sleeper.n.01",
        "slope.n.01",
        "small_indefinite_quantity.n.01",
        "small_person.n.01",
        "smith.n.10",
        "soapsuds.n.01",
        "social_group.n.01",
        "software.n.01",
        "sole.n.01",
        "solid.n.01",
        "solid.n.03",
        "solution.n.01",
        "somatic_cell.n.01",
        "sound.n.04",
        "spatial_property.n.01",
        "species.n.01",
        "specimen.n.01",
        "speech.n.02",
        "speech_act.n.01",
        "sphere.n.01",
        "spirit.n.01",
        "spirit.n.04",
        "spiritual_being.n.01",
        "splash.n.01",
        "spot.n.05",
        "spot.n.12",
        "spring.n.03",
        "square.n.01",
        "squeeze.n.01",
        "stable_gear.n.01",
        "star.n.03",
        "state.n.02",
        "state_of_matter.n.01",
        "statement.n.01",
        "steel.n.01",
        "steward.n.03",
        "store.n.02",
        "story.n.02",
        "stratum.n.01",
        "structure.n.01",
        "structure.n.03",
        "structural_formula.n.01",
        "structure.n.04",
        "styrene.n.01",
        "subject.n.01",
        "subjugation.n.01",
        "substance.n.01",
        "substance.n.07",
        "substance.n.08",
        "suburb.n.01",
        "sum.n.01",
        "superior_skill.n.01",
        "support.n.03",
        "supporting_structure.n.01",
        "surface.n.02",
        "suspension.n.01",
        "sweetening.n.01",
        "swine.n.01",
        "symbol.n.01",
        "symbol.n.02",
        "synapsid.n.01",
        "synthetic.n.01",
        "synthetic_resin.n.01",
        "system.n.01",
        "system.n.06",
        "system_of_measurement.n.01",
        "taste.n.03",
        "taxonomic_group.n.01",
        "temperature_change.n.01",
        "terminal.n.01",
        "test.n.05",
        "text.n.01",
        "texture.n.01",
        "theory.n.01",
        "thing.n.04",
        "thing.n.08",
        "thing.n.12",
        "thinker.n.02",
        "thinking.n.01",
        "thoroughfare.n.01",
        "toecap.n.01",
        "top.n.01",
        "top.n.02",
        "topping.n.01",
        "tract.n.01",
        "trade.n.02",
        "traffic.n.01",
        "transaction.n.01",
        "transducer.n.01",
        "transgression.n.01",
        "transparent_substance.n.01",
        "transportation.n.02",
        "traveler.n.01",
        "triangle.n.01",
        "trouble.n.03",
        "tube.n.01",
        "type.n.04",
        "underbrush.n.01",
        "ungulate.n.01",
        "unicameral_script.n.01",
        "union_representative.n.01",
        "unit.n.02",
        "unit.n.03",
        "unit.n.05",
        "unit_of_measurement.n.01",
        "universe.n.01",
        "unreality.n.01",
        "unsoundness.n.01",
        "unwelcome_person.n.01",
        "upper_class.n.01",
        "upper_surface.n.01",
        "user.n.01",
        "utility.n.06",
        "valuable.n.01",
        "vapor.n.01",
        "vascular_system.n.01",
        "vault.n.03",
        "vegetation.n.01",
        "vehicular_traffic.n.01",
        "veranda.n.01",
        "vertical_surface.n.01",
        "vicinity.n.01",
        "village.n.02",
        "vinyl_polymer.n.01",
        "visual_communication.n.01",
        "visual_percept.n.01",
        "visual_perception.n.01",
        "visual_property.n.01",
        "visual_signal.n.01",
        "vital_principle.n.01",
        "vogue.n.01",
        "volatile_storage.n.01",
        "ware.n.01",
        "waste.n.01",
        "watercourse.n.03",
        "way.n.06",
        "wealth.n.03",
        "weave.n.01",
        "weightlift.n.01",
        "whole.n.01",
        "whole.n.02",
        "window.n.08",
        "woman.n.01",
        "work.n.01",
        "work.n.02",
        "worker.n.01",
        "workman.n.01",
        "workplace.n.01",
        "writing.n.02",
        "writing.n.04",
        "written_communication.n.01",
        "written_symbol.n.01",
        "wrongdoer.n.01",
        "wrongdoing.n.02",
        "yard.n.09",
        "zone.n.01",
    }
)


def generate_all_hypernyms_with_exclusions(
    synset: str | Synset,
    excluded: set[str] | str = EXCLUDED_HYPERNYMS,
    include_self_synset: bool = True,
) -> set[Synset]:
    if synset is None:
        return set()

    if isinstance(synset, str):
        synset = wn.synset(synset)

    return set(
        h
        for hp in synset.hypernym_paths()
        for h in hp
        if (include_self_synset or h != synset) and h.name() not in excluded
    )


@lru_cache(maxsize=10000, typed=True)
def is_hypernym_of(synset: str | Synset, possible_hypernym: str | Synset) -> bool:
    if isinstance(synset, str):
        synset = wn.synset(synset)

    if isinstance(possible_hypernym, str):
        possible_hypernym = wn.synset(possible_hypernym)

    return possible_hypernym in synset.lowest_common_hypernyms(possible_hypernym)


def is_subsynset_of(synset: str | Synset, other_synset: str | Synset) -> bool:
    return is_hypernym_of(synset=synset, possible_hypernym=other_synset)


def symmetric_subsynset_of(synset: str | Synset, other_synset: str | Synset) -> bool:
    return is_hypernym_of(synset=synset, possible_hypernym=other_synset) or is_hypernym_of(
        synset=other_synset, possible_hypernym=synset
    )


def generate_hypernym_to_descendants(
    synsets: Sequence[str] | Sequence[Synset],
) -> dict[str, list[Synset]]:
    if len(synsets) == 0:
        return {}

    if isinstance(synsets[0], str):
        synsets = [wn.synset(s) for s in synsets]

    synsets = set(synsets)
    synsets = [s.name() for s in synsets]

    hypernym_to_descendants = defaultdict(list)
    for s in synsets:
        s = wn.synset(s)
        paths = s.hypernym_paths()
        for hypernym in set(sum(paths, [])):
            hypernym_to_descendants[hypernym.name()].append(s)

    return hypernym_to_descendants


def filter_synsets_to_remove_hyponyms(synsets: Sequence[str] | Sequence[Synset]) -> list[str]:
    if len(synsets) == 0:
        return []

    hyper_to_descs = generate_hypernym_to_descendants(synsets=synsets)

    if isinstance(synsets[0], Synset):
        synsets = [s.name() for s in synsets]

    to_remove = set()
    for synset in synsets:
        descs = hyper_to_descs[synset]
        if len(descs) > 1:
            for desc in descs:
                if desc.name() != synset:
                    to_remove.add(desc.name())

    return list(set(synsets) - to_remove)


def get_all_synsets_in_metadata() -> list[Synset]:
    anns = ObjectMeta.annotation()
    synsets = set(ann["synset"] for ann in anns.values() if "synset" in ann) | set(
        AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET.values()
    )
    synsets = sorted(list(set([wn.synset(s) for s in synsets])), key=lambda s: s.name())
    return synsets


def get_hypernym_to_descendants_for_all_metadata_synsets():
    synsets = get_all_synsets_in_metadata()
    return generate_hypernym_to_descendants(synsets)


@lru_cache(maxsize=10000, typed=True)
def get_hyponyms_of_synset(synset: str | Synset, return_strings: bool) -> set[Synset] | set[str]:
    if isinstance(synset, str):
        synset = wn.synset(synset)

    if return_strings:
        hyps = {synset.name()}
    else:
        hyps = {synset}

    for h in synset.hyponyms():
        hyps.update(
            iter(
                get_hyponyms_of_synset(
                    h,
                    return_strings=return_strings,
                )
            )
        )

    return hyps


def get_hyponyms_of_synsets(
    synsets: Iterable[str] | Iterable[Synset], return_strings: bool
) -> set[Synset] | set[str]:
    hyponyms: set[Synset] | set[str] = set()
    for s in synsets:
        hyponyms.update(iter(get_hyponyms_of_synset(s, return_strings=return_strings)))

    return hyponyms


@cache
def get_singleton_highest_hypernyms():
    highest_hypernyms = Counter()
    for syn in get_all_synsets_in_metadata():
        highest_hypernyms[get_highest_relevant_hypernym(syn)] += 1

    return set([h for h in highest_hypernyms if highest_hypernyms[h] < 2])


def get_highest_relevant_hypernym(
    synset: str | Synset,
    excluded: set[str] | str = EXCLUDED_HYPERNYMS,
) -> Synset:
    if isinstance(synset, str):
        synset = wn.synset(synset)

    for hpath in synset.hypernym_paths():
        for hyp in hpath:
            if hyp.name() not in excluded:
                return hyp.name()

    return synset.name()  # return self if no non-excluded hypernyms


# ------------------------------------------------------------------------------------------
# Receptacle synset filtering with inclusion/exclusion rules
# ------------------------------------------------------------------------------------------

# Synsets to include directly (exact matches)
RECEPTACLE_INCLUDE_SYNSETS: set[str] = frozenset(
    {
        # Tableware categories - things you can put objects in/on
        "flatware.n.01",  # cutlery holder, utensil container
        "glassware.n.01",  # glasses, glass containers
        "dinnerware.n.01",  # plates, bowls, dining items
        "service.n.09",  # set of dishes
        "gold_plate.n.01",  # gilded dishware
        "silver_plate.n.01",  # silverware service
        "crockery.n.01",  # earthenware dishes
        "place_mat.n.01",  # table mats
        "coaster.n.03",  # drink coasters
        # Additional container-like objects
        "tray.n.01",  # serving trays, breakfast trays
        "saucer.n.02",  # small plates for cups
        "platter.n.01",  # serving platters
        # Storage containers
        "jar.n.01",  # glass/ceramic jars
        "canister.n.02",  # kitchen canisters
        "tin.n.02",  # metal containers
        "case.n.05",  # carrying cases
        # Kitchen/household
        "baking_dish.n.01",  # casserole dishes
        "mixing_bowl.n.01",  # large mixing bowls
        "salad_bowl.n.01",  # salad bowls
        "serving_dish.n.01",  # serving dishes
        # Organization items
        "caddy.n.02",  # small organizational containers
        "bin.n.01",  # storage bins
    }
)

# Synsets whose hyponyms should be included (with exceptions)
RECEPTACLE_HYPERNYM_INCLUDE_WITH_EXCLUSIONS: dict[str, set[str]] = {
    "box.n.01": set(),  # all boxes are valid receptacles
    "receptacle.n.01": {"beehive.n.04"},  # all receptacles except beehives
    "pan.n.03": set(),  # all pans
    "vessel.n.03": {
        "ladle.n.01",  # too small/deep for placing
        "bathtub.n.01",  # too large
        "boiler.n.01",  # industrial equipment
        "tank.n.02",  # too large
        "bedpan.n.01",  # sanitary item
    },
    "dish.n.01": set(),  # all dishes
    "basket.n.01": set(),  # all baskets
    "glass.n.02": set(),  # drinking glasses
    "workbasket.n.01": set(),  # sewing/craft baskets
}


@cache
def _get_all_valid_receptacle_synsets() -> set[str]:
    """
    Build the complete set of valid receptacle synsets based on inclusion rules.
    Results are cached for efficiency.

    Returns:
        Set of synset names that are valid receptacles.
    """
    valid_synsets: set[str] = set()

    # Add directly included synsets
    valid_synsets.update(RECEPTACLE_INCLUDE_SYNSETS)

    # Add hyponyms of included hypernyms (with exclusions)
    for hypernym_name, exclusions in RECEPTACLE_HYPERNYM_INCLUDE_WITH_EXCLUSIONS.items():
        hyponyms = get_hyponyms_of_synset(hypernym_name, return_strings=True)
        valid_synsets.update(hyponyms - exclusions)

    return valid_synsets


def is_valid_receptacle_synset(synset: str | Synset) -> bool:
    """
    Check if a synset is a valid receptacle based on inclusion/exclusion rules.

    The cached valid set already contains all hyponyms of included hypernyms,
    so a simple set membership check is sufficient.

    Args:
        synset: A WordNet synset or synset name string

    Returns:
        True if the synset is a valid receptacle type
    """
    if synset is None:
        return False

    if isinstance(synset, Synset):
        synset = synset.name()

    return synset in _get_all_valid_receptacle_synsets()


def get_valid_receptacle_uids() -> dict[str, dict]:
    """
    Get all asset UIDs that are valid receptacles based on synset filtering.

    Returns:
        Dict mapping UID to annotation dict for valid receptacle assets.
    """
    from molmo_spaces.utils.object_metadata import ObjectMeta

    valid_uids = {}

    for uid, anno in ObjectMeta.annotation().items():
        synset = anno.get("synset")
        if synset and is_valid_receptacle_synset(synset):
            # Also check that it's actually a receptacle
            if anno.get("receptacle", False):
                valid_uids[uid] = anno

    return valid_uids


# ---------------------------------------------------------------------------
# Pickupable asset category exclusions
#
# Two mechanisms filter non-everyday-object categories from the pickupable
# asset pool:
#
#   1. Hypernym-based: any synset that is a descendant (hyponym) of an
#      excluded hypernym is removed, along with the hypernym itself.
#      Covers broad categories like sculptures, scale replicas, signs.
#
#   2. Exact synset: specific synsets whose lemma is too generic or
#      non-physical are removed, but their hyponyms are kept.
# ---------------------------------------------------------------------------

PICKUPABLE_EXCLUDED_CATEGORY_HYPERNYMS: dict[str, str] = {
    "sculpture.n.01": "sculpture/art object",
    "model.n.04": "scale model/replica",
    "miniature.n.02": "miniature/replica",
    "sign.n.02": "sign/placard",
    "eolith.n.01": "primitive stone implement",
    "paleolith.n.01": "primitive stone implement",
}

PICKUPABLE_EXCLUDED_EXACT_SYNSETS: dict[str, str] = {
    "plaything.n.01": "generic plaything",
    "toy.n.02": "generic toy (non-plaything sense)",
    "popgun.n.01": "toy gun",
    "arrowhead.n.01": "primitive implement",
    "stone.n.02": "building material",
    "emblem.n.01": "visual symbol",
    "logo.n.01": "visual symbol",
}


def canonical_lemma(synset_name: str) -> str:
    """Return the first (most canonical) lemma for a WordNet synset name."""
    return wn.synset(synset_name).lemma_names()[0].replace("_", " ")


def _build_pickupable_category_exclusion_set() -> dict[str, str]:
    """Map every hypernym-excluded synset (including its descendants) to its
    exclusion reason string."""
    excluded: dict[str, str] = {}
    for hypernym, reason in PICKUPABLE_EXCLUDED_CATEGORY_HYPERNYMS.items():
        for s in get_hyponyms_of_synset(hypernym, return_strings=True):
            excluded.setdefault(s, reason)
    return excluded


@cache
def _pickupable_synset_to_uids() -> dict[str, list[str]]:
    """Cached mapping of synset -> UIDs for all valid pickupable assets
    (before category exclusions)."""
    uids = get_valid_pickupable_obja_uids_excluding_benchmark()
    annotations = ObjectMeta.annotation()

    synset_to_uids: dict[str, list[str]] = defaultdict(list)
    for uid in uids:
        anno = annotations.get(uid)
        if anno is None:
            continue
        synset = anno.get("synset")
        if synset is None:
            continue
        synset_to_uids[synset].append(uid)

    return dict(synset_to_uids)


@cache
def _pickupable_class_ranking() -> list[tuple[str, list[str]]]:
    """Ranked list of kept pickupable asset classes (cached).

    Returns ``(synset, uids)`` tuples sorted descending by asset count,
    after applying both hypernym-based and exact-synset exclusions.
    """
    synset_to_uids = _pickupable_synset_to_uids()
    hypernym_excluded = _build_pickupable_category_exclusion_set()

    kept_synsets = [
        s
        for s in synset_to_uids
        if s not in hypernym_excluded and s not in PICKUPABLE_EXCLUDED_EXACT_SYNSETS
    ]
    kept_synsets.sort(key=lambda s: len(synset_to_uids[s]), reverse=True)
    return [(s, synset_to_uids[s]) for s in kept_synsets]


VALID_PICKUPABLE_OBJA_UIDS_PATH = (
    "/weka/prior/datasets/robomolmo/asset_utility_refs/valid_pickupable_obja_uids.txt"
)

BENCHMARK_BLACKLIST_UIDS_PATH = (
    "/weka/prior/datasets/robomolmo/asset_utility_refs/benchmark_blacklist_uids.txt"
)


def _load_uid_list(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_valid_pickupable_obja_uids(debug: bool = False) -> list[str]:
    """
    Get all objaverse asset UIDs that are pickable (have valid grasp files).

    Checks for cached file at VALID_PICKUPABLE_OBJA_UIDS_PATH first to avoid
    expensive computation. If not found, computes and returns the list.

    Args:
        debug: If True, prints 20 random samples with their short descriptions.

    Returns:
        List of UIDs for valid pickupable assets.
    """
    import os

    if os.path.exists(VALID_PICKUPABLE_OBJA_UIDS_PATH):
        with open(VALID_PICKUPABLE_OBJA_UIDS_PATH) as f:
            uid_list = [line.strip() for line in f if line.strip()]
        if debug:
            print(f"\n=== Loaded {len(uid_list)} pickupable UIDs from cache ===\n")
        return uid_list

    from molmo_spaces.utils.grasps import has_valid_pickup_grasps
    from molmo_spaces.utils.object_metadata import ObjectMeta

    valid_uids = {}

    for uid, anno in ObjectMeta.annotation().items():
        if has_valid_pickup_grasps(uid):
            valid_uids[uid] = anno

    if debug:
        import random

        sample_size = min(20, len(valid_uids))
        sample_uids = random.sample(list(valid_uids.keys()), sample_size)

        print(f"\n=== Pickable Objaverse Assets ({len(valid_uids)} total) ===")
        print(f"Random sample of {sample_size}:\n")

        for uid in sample_uids:
            anno = valid_uids[uid]
            short_descs = list(anno.get("description_short", {}).values())
            desc_str = short_descs[0] if short_descs else anno.get("category", "N/A")
            print(f"  {uid}: {desc_str}")

        print()

    return list(valid_uids.keys())


def get_valid_pickupable_obja_uids_excluding_benchmark(debug: bool = False) -> list[str]:
    """Get pickupable objaverse UIDs with benchmark assets excluded.

    Loads the benchmark blacklist from BENCHMARK_BLACKLIST_UIDS_PATH
    (generated by scripts/roseh/extract_benchmark_assets.py) and removes
    any UIDs that appear in any bench_v3 benchmark as a pickup or placement
    asset.

    Args:
        debug: If True, prints how many assets were filtered and samples
            of what was removed.

    Returns:
        List of UIDs for valid pickupable assets not used in any benchmark.
    """
    all_uids = get_valid_pickupable_obja_uids(debug=False)
    blacklist = set(_load_uid_list(BENCHMARK_BLACKLIST_UIDS_PATH))

    original_count = len(all_uids)
    filtered = [uid for uid in all_uids if uid not in blacklist]
    removed_count = original_count - len(filtered)

    if debug:
        print(
            f"\n=== Benchmark blacklist filtering ===\n"
            f"  {original_count} pickupable UIDs total\n"
            f"  {len(blacklist)} UIDs in benchmark blacklist\n"
            f"  {removed_count} removed (appeared in both)\n"
            f"  {len(filtered)} remaining after filtering\n"
        )

    return filtered
