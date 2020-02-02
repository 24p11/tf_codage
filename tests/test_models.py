from tf_codage.models import FullTextBert, TFCamembertModel
from tf_codage import models
from transformers import CamembertConfig
from transformers import CamembertTokenizer
import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose
from tempfile import mkdtemp

wiki_text = ["""
Paris [pa.ʁi]a Écouter est la ville la plus peuplée et la capitale de la France.

Elle se situe au cœur d'un vaste bassin sédimentaire aux sols fertiles et au climat tempéré, le bassin parisien, sur une boucle de la Seine, entre les confluents de celle-ci avec la Marne et l'Oise. Paris est également le chef-lieu de la région Île-de-France et le centre de la métropole du Grand Paris, créée en 2016. Elle est divisée en arrondissements, comme les villes de Lyon et de Marseille, au nombre de vingt. Administrativement, la ville constitue depuis le 1er janvier 2019 une collectivité à statut particulier nommée « Ville de Paris » (auparavant, elle était à la fois une commune et un département). L'État y dispose de prérogatives particulières exercées par le préfet de police de Paris. La ville a connu de profondes transformations sous le Second Empire dans les décennies 1850 et 1860 à travers d'importants travaux consistant notamment au percement de larges avenues, places et jardins et la construction de nombreux édifices, dirigés par le baron Haussmann.

La ville de Paris comptait 2,19 millions d'habitants au 1er janvier 2016. Ses habitants sont appelés parisiens. L'agglomération parisienne s’est largement développée au cours du XXe siècle, rassemblant 10,73 millions d'habitants au 1er janvier 2016, et son aire urbaine (l'agglomération et la couronne périurbaine) comptait 12,57 millions d'habitants. L'agglomération parisienne est ainsi la plus peuplée de France, elle est la quatrième du continent européen et la 32e plus peuplée du monde au 1er janvier 2019.

La position de Lutèce, sur l'île aujourd'hui nommée l'île de la Cité, permettant le franchissement du grand fleuve navigable qu'est la Seine par une voie reliant le Nord et le Sud des Gaules, en fait dès l'Antiquité une cité importante, capitale des Parisii, puis lieu de séjour d'un empereur romain. Sa position au centre du territoire contrôlé par les rois francs la fait choisir comme capitale de la France à la place de Tournai. Située au cœur d'un territoire agricole fertile avec un climat humide et doux, Paris devient une des principales villes de France au cours du Xe siècle, avec des palais royaux, de riches abbayes et une cathédrale. Au cours du XIIe siècle, avec l'université de Paris, la cité devient un des premiers foyers en Europe pour l’enseignement et les arts. Le pouvoir royal se fixant dans cette ville, son importance économique et politique ne cesse de croître. Ainsi, au début du XIVe siècle, Paris est l'une des villes les plus importantes du monde chrétien. Au XVIIe siècle, elle est la capitale de la principale puissance politique européenne ; au XVIIIe siècle, l'un des plus grands centres culturels de l’Europe ; et au XIXe siècle, la capitale des arts et des plaisirs. Du XVIe siècle au XXe siècle, Paris a été la capitale de l'Empire colonial français. Paris joue donc un rôle de tout premier plan dans l'histoire de l'Europe et du monde depuis des siècles.

Paris symbolise la culture française. En 2017 elle est classée comme étant la ville la plus élégante au monde. Elle abrite de nombreux monuments, surnommée la Ville Lumière, elle attire en 2019 près de 34 millions de visiteurs, ce qui en fait une des capitales les plus visitées au monde. Elle abrite le musée d'art le plus grand et le plus visité au monde. Dans le secteur du luxe le premier et le troisième groupe mondial ont leur siège à Paris qui est également la ville qui en 2018 compte le plus de palaces au monde. À Paris se déroule tous les ans une semaine internationale de la mode. C'est dans cette ville qu'ont exercé et qu'exercent des couturiers de renommée mondiale et des marques française du luxe sont connues sur tous les continents. Dans le secteur de la haute gastronomie Paris est la ville qui compte le plus grand nombre de meilleures tables au monde. La capitale française n'est jumelée qu'avec une seule autre ville, Rome, ce qui est aussi valable dans l'autre sens, avec ce slogan « Seul Paris est digne de Rome, seule Rome est digne de Paris ». 

La ville est, avec sa banlieue, la capitale économique de la France. Elle est la première place financière et boursière du pays. En 2017 le quartier d'affaires de La Défense est le plus grand d'Europe et le quatrième dans le monde en terme d'attractivité. Toujours en 2017 la région parisienne accueille plus d'institutions internationales et de sièges sociaux de très grandes entreprises que New York et que Londres. En 2018 elle est le siège de deux des dix plus grandes banques mondiales. Elle est également le siège d'organismes européens tels l'Autorité européenne des marchés financiers et l'Autorité bancaire européenne, et d'organismes internationaux tels l'UNESCO, l'OCDE, l'ICC, le GAFI. La région parisienne est l'une des plus riches régions d'Europe. En 1989, elle a été désignée capitale européenne de la culture, et en 2017, capitale européenne de l'innovation.

La densité de ses réseaux ferroviaire, autoroutier et de ses structures aéroportuaires en font un point de convergence pour les transports nationaux et internationaux. Cette situation résulte d'une longue évolution, en particulier des conceptions centralisatrices des monarchies et des républiques, qui donnent un rôle considérable à la capitale dans le pays et tendent à y concentrer les institutions. Depuis les années 1960, les politiques gouvernementales oscillent toutefois entre déconcentration et décentralisation. La macrocéphalie dont est atteinte la ville se concrétise par la convergence de la plupart des réseaux routiers et ferroviaires du pays en son centre et des écarts démographiques et économiques disproportionnés entre la capitale et la province : près de 19 % de la population française vit dans l'aire urbaine de Paris.

Le club de football du Paris Saint-Germain et celui de rugby à XV du Stade français sont basés à Paris. Le Stade de France, enceinte de 80 000 places construite pour la Coupe du monde de football de 1998, est situé au nord de la capitale, dans la commune voisine de Saint-Denis. Paris, qui accueille chaque année le tournoi du Grand Chelem de tennis de Roland Garros, a organisé les Jeux olympiques en 1900 puis en 1924 et deviendra en 2024 la deuxième ville avec Londres à les avoir accueillis trois fois. C'est à Paris que se déroule tous les ans deux des plus prestigieuses courses hippiques au monde. Paris accueille également de nombreuses compétitions internationales et — depuis 1975 — l'arrivée du Tour de France, troisième événement sportif le plus suivi au monde. 
""",
"""
Lyon (prononcé /ljɔ̃/3 ou /liɔ̃/ Écouter4) est une commune française située dans le sud-est de la France au confluent du Rhône et de la Saône. Siège du conseil de la métropole de Lyon5, elle est le chef-lieu de l'arrondissement de Lyon, de la circonscription départementale du Rhône et de la région Auvergne-Rhône-Alpes. Le gentilé est Lyonnais.

Lyon a une situation de carrefour géographique du pays, au nord du couloir naturel de la vallée du Rhône (qui s'étend de Lyon à Marseille). Située entre le Massif central à l'ouest et le massif alpin à l'est, la ville de Lyon occupe une position stratégique dans la circulation nord-sud en Europe. Ancienne capitale des Gaules du temps de l'Empire romain, Lyon est le siège d'un archevêché dont le titulaire porte le titre de primat des Gaules. Lyon devint une ville très commerçante et une place financière de premier ordre à la Renaissance. Sa prospérité économique a été portée successivement par la soierie, puis par l'apparition des industries notamment textiles, chimiques, et plus récemment, par l'industrie de l'image.

Lyon, historiquement ville industrielle, a accueilli au sud de la ville de nombreuses industries pétrochimiques le long du Rhône, nommé le couloir de la chimie. Après le départ et la fermeture des industries textiles, Lyon s'est progressivement recentrée sur les secteurs d'activité de techniques de pointe, telles que la pharmacie et les biotechnologies. Lyon est également la deuxième ville étudiante de France, avec quatre universités et plusieurs grandes écoles. Enfin, la ville a conservé un patrimoine architectural important allant de l'époque romaine au XXe siècle en passant par la Renaissance et, à ce titre, les quartiers du Vieux Lyon, de la colline de Fourvière, de la Presqu'île et des pentes de la Croix-Rousse sont inscrits sur la liste du patrimoine mondial de l'UNESCO.

Par sa population, Lyon constitue la troisième commune de France, avec 516 092 habitants au dernier recensement de 2017. Lyon est ville-centre de la 2e unité urbaine de France, laquelle comptait 1 659 001 habitants en 2017 et de la 2e aire urbaine (2 326 223 habitants en 2017) de France. Elle est la préfecture de la région Auvergne-Rhône-Alpes et le siège de la métropole de Lyon, qui rassemble 59 communes et 1 385 927 habitants6 en 2017. La ville de Lyon exerce une attractivité d'importance nationale et européenne. Son importance dans les domaines culturels, bancaires, financiers, commerciaux, technologiques, pharmaceutiques, ou encore les arts et les divertissements font de la ville de Lyon une ville mondiale de rang « Beta- » selon le classement GaWC en 2016, comparable à Seattle, Amman ou Anvers7. Lyon est également le siège d'Interpol depuis 1989. 
"""]

def test_full_text_bert_layer_id():
    """test selection of layer for classfication"""
    config = CamembertConfig(max_position_embeddings=514)
    model = FullTextBert(config, cls_token=5, sep_token=6, layer_id=-2)
    
    input_ids = tf.constant(
    [[3, 1, 0, 5, 6],
     [7, 2, 1, 1, 1],
     [0, 1, 0, 3, 3]])
    
    outputs, mask = model(input_ids)
    assert outputs.shape == (3, 10, 512, 768)
    assert mask is None
    
def test_full_text_bert():
    
    config = CamembertConfig(max_position_embeddings=514)
    
    model = FullTextBert(config, cls_token=5, sep_token=6)
    
    input_ids = tf.constant(
    [[3, 1, 0, 5, 6],
     [7, 2, 1, 1, 1],
     [0, 1, 0, 3, 3]])
    
    outputs, mask = model(input_ids)
    assert outputs.shape == (3, 10, 512, 768)
    assert mask is None
    
    # test with dict input
    outputs_dict, _ = model({'input_ids': input_ids})
    assert outputs_dict.shape == (3, 10, 512, 768)
    assert (outputs_dict.numpy() == outputs.numpy()).all()
    
    # test with attention mask
    outputs_attention, _ = model({'input_ids': input_ids,
                              'attention_mask': tf.ones((3, 5), tf.int32)})
    assert outputs.shape == (3, 10, 512, 768)
    # because of the padding
    assert not (outputs_attention.numpy() == outputs.numpy()).all()
    
def test_full_text_bert_attention_mask():
    
    config = CamembertConfig(max_position_embeddings=514)
    model = FullTextBert(config, cls_token=5, sep_token=6)
    
    input1 = tf.constant([[3, 4, 8, 7]])
    attention_mask1 = tf.constant([[1, 1, 1, 1]])
    input2 = tf.constant([[3, 4, 8, 7, 5]])
    attention_mask2a = tf.constant([[1, 1, 1, 1, 1]])
    attention_mask2b = tf.constant([[1, 1, 1, 1, 0]])
    
    
    out1, mask1 = model({'input_ids': input1,
                  'attention_mask': attention_mask1})
    out2a, mask2a = model({'input_ids': input2,
                  'attention_mask': attention_mask2a})
    out2b, mask2b = model({'input_ids': input2,
                  'attention_mask': attention_mask2b})
    
    
    # first 4 tokens should not be influenced by the last one if the mask set
    assert_allclose(out1.numpy()[0, 0, :5, :], out2b.numpy()[0, 0, :5, :])
    
    # last token should differ 
    assert not np.allclose(out1.numpy()[0, 0, 5, :], out2b.numpy()[0, 0, 5, :])
    
    # if the mask is not set, the last token my influence the rest
    assert not np.allclose(out1.numpy()[0, 0, :5, :], out2a.numpy()[0, 0, :5, :])
    
    # test masks
    expected_mask = np.zeros((1, 10, 512))
    expected_mask[0, 0, 1:5] = 1
    # special tokens are not masked
    expected_mask[:, :,  0] = 1
    expected_mask[:, :, -1] = 1
    assert np.equal(mask1.numpy(), expected_mask).all()
    assert np.equal(mask2b.numpy(), expected_mask).all()
    
    expected_mask[0, 0, 5] = 1
    assert_allclose(mask2a.numpy(), expected_mask)


def test_full_text_bert_compare():
    """Compare full text bert with batches of standard camembert"""
    
    model_dir = mkdtemp()
     
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    config = CamembertConfig.from_pretrained("camembert-base")

    cls_token = tokenizer.cls_token_id
    sep_token = tokenizer.sep_token_id
    max_batches = 2
    
    multi_token = [tokenizer.encode(
        text,
        pad_to_max_length=True, 
        max_length=max_batches * 510, 
        add_special_tokens=False) for text in  wiki_text]
    
    def split_tokens(t):
        return [([cls_token] + t[i * 510:(i+1)*510] + [sep_token]) for i in range(max_batches)]
    
    bert_inputs = [split_tokens(tok_ids) for tok_ids in multi_token] 
    
    single_model = TFCamembertModel(config)
    out_bert = np.array([single_model(np.array(b))[0].numpy() for b in bert_inputs])

    single_model.save_pretrained(model_dir)
    fulltext_model = FullTextBert.from_pretrained(
        model_dir, cls_token=cls_token, sep_token=sep_token, max_batches=max_batches) 
    
    
    out_full_bert, _ = fulltext_model(np.array(multi_token))
    
    assert_allclose(out_bert, out_full_bert.numpy(), atol=1e-4)


def test_mean_masked_pooling_layer():
    """Test MeanMaskPoolingLayer"""
    
    n_batch = 4
    n_tokens = 16
    n_splits = 2
    n_hidden = 2
    hidden_inputs = np.random.randn(n_batch, n_tokens, n_splits, n_hidden).astype(np.float32)
    mask = np.ones((n_batch, n_tokens, n_splits))
    
    config = models.FullTextConfig(pool_size=32, pool_strides=1)
    pooling = models.MeanMaskedPooling(config)
    
    output = pooling(hidden_inputs, mask).numpy()
    expected = hidden_inputs.reshape(n_batch, n_tokens * n_splits, 1, n_hidden).mean(1)
    assert_allclose(expected, output)
    
    # test with different mask
    mask[:, 8:, :] = 0
    output = pooling(hidden_inputs, mask).numpy()
    expected = hidden_inputs[:, :8, :, :].reshape(n_batch, 8 * n_splits, 1, n_hidden).mean(1)
    assert_allclose(expected, output)
    
    # test non-overlapping strides
    config = models.FullTextConfig(pool_size=8, pool_strides=8)
    pooling = models.MeanMaskedPooling(config)
    mask = np.ones((n_batch, n_tokens, n_splits))
    
    output = pooling(hidden_inputs, mask).numpy()
    
    assert output.shape == (n_batch, 4, n_hidden)
    expected = hidden_inputs.reshape(n_batch, n_tokens * n_splits // 8, 8, n_hidden).mean(2)
    assert_allclose(expected, output)
    
    # test shape with overlapping strides
    config = models.FullTextConfig(pool_size=8, pool_strides=4)
    pooling = models.MeanMaskedPooling(config)
    
    output = pooling(hidden_inputs, mask).numpy()
    
    assert output.shape == (n_batch, 7, n_hidden)
    
def test_max_masked_pooling_layer():
    """Test MaxMaskPoolingLayer"""
    
    n_batch = 4
    n_tokens = 16
    n_splits = 2
    n_hidden = 2
    hidden_inputs = np.random.randn(n_batch, n_tokens, n_splits, n_hidden).astype(np.float32)
    mask = np.ones((n_batch, n_tokens, n_splits))
    
    config = models.FullTextConfig(pool_size=32, pool_strides=1)
    pooling = models.MaxMaskedPooling(config)
    
    output = pooling(hidden_inputs, mask).numpy()
    expected = hidden_inputs.reshape(n_batch, n_tokens * n_splits, 1, n_hidden).max(1)
    assert_allclose(expected, output)
    
    # test with different mask
    mask[:, 8:, :] = 0
    output = pooling(hidden_inputs, mask).numpy()
    expected = hidden_inputs[:, :8, :, :].reshape(n_batch, 8 * n_splits, 1, n_hidden).max(1)
    assert_allclose(expected, output)
