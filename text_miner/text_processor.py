from collections import Counter, OrderedDict

import pycld2 as cld2
import spacy
import numpy as np
from sklearn import preprocessing
from nltk.corpus import stopwords

ISO_STANDARDS = {"639_1": 0, "639_2/t": 1, "639_2/b": 2}
ISO_LANGUAGES = {"abkhaz": ['ab', 'abk', 'abk'], "afar": ['aa', 'aar', 'aar'],
                 "afrikaans": ['af', 'afr', 'afr'], "akan": ['ak', 'aka', 'aka'],
                 "albanian": ['sq', 'sqi', 'alb'], "amharic": ['am', 'amh', 'amh'],
                 "arabic": ['ar', 'ara', 'ara'], "aragonese": ['an', 'arg', 'arg'],
                 "armenian": ['hy', 'hye', 'arm'], "assamese": ['as', 'asm', 'asm'],
                 "avaric": ['av', 'ava', 'ava'], "avestan": ['ae', 'ave', 'ave'],
                 "aymara": ['ay', 'aym', 'aym'], "azerbaijani": ['az', 'aze', 'aze'],
                 "bambara": ['bm', 'bam', 'bam'], "bashkir": ['ba', 'bak', 'bak'],
                 "basque": ['eu', 'eus', 'baq'], "belarusian": ['be', 'bel', 'bel'],
                 "bengali": ['bn', 'ben', 'ben'], "bihari": ['bh', 'bih', 'bih'],
                 "bislama": ['bi', 'bis', 'bis'], "bosnian": ['bs', 'bos', 'bos'],
                 "breton": ['br', 'bre', 'bre'], "bulgarian": ['bg', 'bul', 'bul'],
                 "burmese": ['my', 'mya', 'bur'], "catalan": ['ca', 'cat', 'cat'],
                 "chamorro": ['ch', 'cha', 'cha'], "chechen": ['ce', 'che', 'che'],
                 "chichewa": ['ny', 'nya', 'nya'], "chinese": ['zh', 'zho', 'chi'],
                 "chuvash": ['cv', 'chv', 'chv'], "cornish": ['kw', 'cor', 'cor'],
                 "corsican": ['co', 'cos', 'cos'], "cree": ['cr', 'cre', 'cre'],
                 "croatian": ['hr', 'hrv', 'hrv'], "czech": ['cs', 'ces', 'cze'],
                 "danish": ['da', 'dan', 'dan'], "divehi": ['dv', 'div', 'div'],
                 "dutch": ['nl', 'nld', 'dut'], "dzongkha": ['dz', 'dzo', 'dzo'],
                 "english": ['en', 'eng', 'eng'], "esperanto": ['eo', 'epo', 'epo'],
                 "estonian": ['et', 'est', 'est'], "ewe": ['ee', 'ewe', 'ewe'],
                 "faroese": ['fo', 'fao', 'fao'], "fijian": ['fj', 'fij', 'fij'],
                 "finnish": ['fi', 'fin', 'fin'], "french": ['fr', 'fra', 'fre'],
                 "fula": ['ff', 'ful', 'ful'], "galician": ['gl', 'glg', 'glg'],
                 "georgian": ['ka', 'kat', 'geo'], "german": ['de', 'deu', 'ger'],
                 "greek ": ['el', 'ell', 'gre'], "guaraní": ['gn', 'grn', 'grn'],
                 "gujarati": ['gu', 'guj', 'guj'], "haitian": ['ht', 'hat', 'hat'],
                 "hausa": ['ha', 'hau', 'hau'], "hebrew": ['he', 'heb', 'heb'],
                 "herero": ['hz', 'her', 'her'], "hindi": ['hi', 'hin', 'hin'],
                 "hiri motu": ['ho', 'hmo', 'hmo'], "hungarian": ['hu', 'hun', 'hun'],
                 "interlingua": ['ia', 'ina', 'ina'], "indonesian": ['id', 'ind', 'ind'],
                 "interlingue": ['ie', 'ile', 'ile'], "irish": ['ga', 'gle', 'gle'],
                 "igbo": ['ig', 'ibo', 'ibo'], "inupiaq": ['ik', 'ipk', 'ipk'],
                 "ido": ['io', 'ido', 'ido'], "icelandic": ['is', 'isl', 'ice'],
                 "italian": ['it', 'ita', 'ita'], "inuktitut": ['iu', 'iku', 'iku'],
                 "japanese": ['ja', 'jpn', 'jpn'], "javanese": ['jv', 'jav', 'jav'],
                 "kalaallisut": ['kl', 'kal', 'kal'], "kannada": ['kn', 'kan', 'kan'],
                 "kanuri": ['kr', 'kau', 'kau'], "kashmiri": ['ks', 'kas', 'kas'],
                 "kazakh": ['kk', 'kaz', 'kaz'], "khmer": ['km', 'khm', 'khm'],
                 "kikuyu": ['ki', 'kik', 'kik'], "kinyarwanda": ['rw', 'kin', 'kin'],
                 "kyrgyz": ['ky', 'kir', 'kir'], "komi": ['kv', 'kom', 'kom'],
                 "kongo": ['kg', 'kon', 'kon'], "korean": ['ko', 'kor', 'kor'],
                 "kurdish": ['ku', 'kur', 'kur'], "kwanyama": ['kj', 'kua', 'kua'],
                 "latin": ['la', 'lat', 'lat'], "luxembourgish": ['lb', 'ltz', 'ltz'],
                 "ganda": ['lg', 'lug', 'lug'], "limburgish": ['li', 'lim', 'lim'],
                 "lingala": ['ln', 'lin', 'lin'], "lao": ['lo', 'lao', 'lao'],
                 "lithuanian": ['lt', 'lit', 'lit'], "luba-katanga": ['lu', 'lub', 'lub'],
                 "latvian": ['lv', 'lav', 'lav'], "manx": ['gv', 'glv', 'glv'],
                 "macedonian": ['mk', 'mkd', 'mac'], "malagasy": ['mg', 'mlg', 'mlg'],
                 "malay": ['ms', 'msa', 'may'], "malayalam": ['ml', 'mal', 'mal'],
                 "maltese": ['mt', 'mlt', 'mlt'], "māori": ['mi', 'mri', 'mao'],
                 "marathi": ['mr', 'mar', 'mar'], "marshallese": ['mh', 'mah', 'mah'],
                 "mongolian": ['mn', 'mon', 'mon'], "nauruan": ['na', 'nau', 'nau'],
                 "navajo": ['nv', 'nav', 'nav'], "northern ndebele": ['nd', 'nde', 'nde'],
                 "nepali": ['ne', 'nep', 'nep'], "ndonga": ['ng', 'ndo', 'ndo'],
                 "norwegian bokmål": ['nb', 'nob', 'nob'],
                 "norwegian nynorsk": ['nn', 'nno', 'nno'],
                 "norwegian": ['no', 'nor', 'nor'], "nuosu": ['ii', 'iii', 'iii'],
                 "southern ndebele": ['nr', 'nbl', 'nbl'], "occitan": ['oc', 'oci', 'oci'],
                 "ojibwe": ['oj', 'oji', 'oji'], "old bulgarian": ['cu', 'chu', 'chu'],
                 "oromo": ['om', 'orm', 'orm'], "oriya": ['or', 'ori', 'ori'],
                 "ossetian": ['os', 'oss', 'oss'], "panjabi": ['pa', 'pan', 'pan'],
                 "pāli": ['pi', 'pli', 'pli'], "persian": ['fa', 'fas', 'per'],
                 "polish": ['pl', 'pol', 'pol'], "pashto": ['ps', 'pus', 'pus'],
                 "portuguese": ['pt', 'por', 'por'], "quechua": ['qu', 'que', 'que'],
                 "romansh": ['rm', 'roh', 'roh'], "kirundi": ['rn', 'run', 'run'],
                 "romanian": ['ro', 'ron', 'rum'], "russian": ['ru', 'rus', 'rus'],
                 "sanskrit": ['sa', 'san', 'san'], "sardinian": ['sc', 'srd', 'srd'],
                 "sindhi": ['sd', 'snd', 'snd'], "northern sami": ['se', 'sme', 'sme'],
                 "samoan": ['sm', 'smo', 'smo'], "sango": ['sg', 'sag', 'sag'],
                 "serbian": ['sr', 'srp', 'srp'], "gaelic": ['gd', 'gla', 'gla'],
                 "shona": ['sn', 'sna', 'sna'], "sinhala": ['si', 'sin', 'sin'],
                 "slovak": ['sk', 'slk', 'slo'], "slovene": ['sl', 'slv', 'slv'],
                 "somali": ['so', 'som', 'som'], "southern sotho": ['st', 'sot', 'sot'],
                 "spanish": ['es', 'spa', 'spa'], "sundanese": ['su', 'sun', 'sun'],
                 "swahili": ['sw', 'swa', 'swa'], "swati": ['ss', 'ssw', 'ssw'],
                 "swedish": ['sv', 'swe', 'swe'], "tamil": ['ta', 'tam', 'tam'],
                 "telugu": ['te', 'tel', 'tel'], "tajik": ['tg', 'tgk', 'tgk'],
                 "thai": ['th', 'tha', 'tha'], "tigrinya": ['ti', 'tir', 'tir'],
                 "tibetan": ['bo', 'bod', 'tib'], "turkmen": ['tk', 'tuk', 'tuk'],
                 "tagalog": ['tl', 'tgl', 'tgl'], "tswana": ['tn', 'tsn', 'tsn'],
                 "tonga": ['to', 'ton', 'ton'], "turkish": ['tr', 'tur', 'tur'],
                 "tsonga": ['ts', 'tso', 'tso'], "tatar": ['tt', 'tat', 'tat'],
                 "twi": ['tw', 'twi', 'twi'], "tahitian": ['ty', 'tah', 'tah'],
                 "uyghur": ['ug', 'uig', 'uig'], "ukrainian": ['uk', 'ukr', 'ukr'],
                 "urdu": ['ur', 'urd', 'urd'], "uzbek": ['uz', 'uzb', 'uzb'],
                 "venda": ['ve', 'ven', 'ven'], "vietnamese": ['vi', 'vie', 'vie'],
                 "volapük": ['vo', 'vol', 'vol'], "walloon": ['wa', 'wln', 'wln'],
                 "welsh": ['cy', 'cym', 'wel'], "wolof": ['wo', 'wol', 'wol'],
                 "western frisian": ['fy', 'fry', 'fry'], "xhosa": ['xh', 'xho', 'xho'],
                 "yiddish": ['yi', 'yid', 'yid'], "yoruba": ['yo', 'yor', 'yor'],
                 "zhuang  chuang": ['za', 'zha', 'zha'], "zulu": ['zu', 'zul', 'zul']}

SPACY_LANGUAGE_MAPPING = {
    # "spanish": "es_core_news_sm",
    "spanish": "es_core_news_md",
    # "english": "en_core_web_sm",
    "english": "en_core_web_md",
    "portuguese": "pt_core_news_sm",
    "italian": "it_core_news_sm",
    "french": "fr_core_news_sm",
    "german": "de_core_news_sm"
}


class TextProcessor:
    def __init__(self, text, iso_standard="639_1", language_name=None, language_iso_code=None, preprocess=True):
        if ISO_STANDARDS.get(iso_standard) is None:
            raise ValueError(f"ISO standard {iso_standard} not recognizable")
        if language_name is not None and language_name not in ISO_LANGUAGES:
            raise ValueError(f"Language {language_name} not recognizable")
        self.text = text
        self._nlp_doc = None

        self.iso_standard = iso_standard
        self.lang = language_name
        self.lang_iso_code = language_iso_code
        if language_name is None and language_iso_code is None:
            self.lang, self.lang_iso_code = self.detect_language(text)
        elif language_name is None:
            self.lang = self._get_lang_from_lang_code(language_iso_code)
        elif language_iso_code is None:
            self.lang_iso_code = self._get_lang_code_from_lang(language_name)

        if preprocess:
            self._nlp_doc = self.make_nlp()

    @staticmethod
    def detect_language(text):
        try:
            # text = bytes(text, 'utf-8').decode('utf-8', 'backslashreplace')
            detected = cld2.detect(text)
            if detected[0]:
                lang = detected[2][0][0].lower()
                lang_code = detected[2][0][1]
            else:
                lang = lang_code = None
        except Exception as err:
            raise Exception("TextProcessor::detect_language: " + str(err))
        return lang, lang_code

    def _get_lang_from_lang_code(self, lang_code):
        idx = ISO_STANDARDS[self.iso_standard]
        for key, value in ISO_LANGUAGES.items():
            if value[idx] == lang_code:
                return key
        raise ValueError(f"ISO language code {lang_code} not recognizable")

    def _get_lang_code_from_lang(self, language_name):
        idx = ISO_STANDARDS[self.iso_standard]
        return ISO_LANGUAGES[language_name][idx]

    def make_nlp(self, disabled_pipes=("parser", "ner", "textcat", "entity_ruler", "merge_entities")):
        lang_to_load = SPACY_LANGUAGE_MAPPING[self.lang]
        nlp = spacy.load(lang_to_load, disable=disabled_pipes)
        return nlp(self.text)

    def _get_tokens_property_list(self, prop_name, check_attribute=(), check_if_false=False):
        if self._nlp_doc is None:
            # raise ValueError("Must execute method make_nlp first for this instance")
            self.make_nlp()
        if check_attribute:
            if check_if_false:
                return [getattr(tok, prop_name) for tok in self._nlp_doc if not any(
                    [getattr(tok, attr) for attr in check_attribute]
                )]
            return [getattr(tok, prop_name) for tok in self._nlp_doc if any(
                [getattr(tok, attr) for attr in check_attribute]
            )]
        return [getattr(tok, prop_name) for tok in self._nlp_doc]

    def get_tokens(self, lowercase=True):
        return self._get_tokens_property_list("lower_" if lowercase else "text")

    def get_lemmas(self):
        return self._get_tokens_property_list("lemma_")

    def get_numerical_tokens(self):
        return self._get_tokens_property_list("text", check_attribute=("is_digit",))

    def get_alpha_tokens(self, lowercase=True):
        return self._get_tokens_property_list("lower_" if lowercase else "text", check_attribute=("is_alpha",))

    def get_urllike_tokens(self):
        return self._get_tokens_property_list("text", check_attribute=("like_url",))

    def get_emaillike_tokens(self):
        return self._get_tokens_property_list("text", check_attribute=("like_email",))

    def get_filtered_dimensions(self):
        attributes = ("is_stop", "is_punct", "is_quote", "is_currency", "is_space", "is_digit")
        features = self._get_tokens_property_list("lemma_", check_attribute=attributes, check_if_false=True)

        # Convert all features to lowercase, because spacy let capitalized words stays intact.
        # Because nltk stopwords definition is more complete than spacy, here we check this.
        features = [f.lower() for f in features if f not in stopwords.words(self.lang)]

        # if self.lang == "spanish":
        # features = [feat for feat in features if len(feat) > 1]
        return features

    def get_bag_of_words(self, order='numerical'):
        cleaned_features = self.get_filtered_dimensions()
        # tf = {}
        # for feature in cleaned_features:
        #     tf[feature] = tf.get(feature, 0) + 1
        tf = Counter(cleaned_features)
        if order != 'numerical':
            return self._order_dict(tf, order_method=order)
        return tf

    @staticmethod
    def _order_dict(dict_, order_method='alphabetical'):
        allowed_methods = ("alphabetical", "numerical")
        if order_method not in allowed_methods:
            all_in_or = " or ".join(allowed_methods)
            raise ValueError(f"Wrong value of norm, {all_in_or} expected")
        ordered_dict = OrderedDict()
        if order_method == allowed_methods[0]:
            ordered_keys = sorted(dict_)
        else:
            ordered_keys = sorted(dict_, key=dict_.get, reverse=True)
        for key in ordered_keys:
            ordered_dict[key] = dict_[key]
        return ordered_dict

    def get_unit_vector(self, norm='l2', order='numerical'):
        if norm not in ('l1', 'l2'):
            raise ValueError("Wrong value of norm, 'l1' or 'l2' expected")
        bog = self.get_bag_of_words(order=order)
        word_list = []
        values_list = []
        if type(bog) == Counter:
            for word, frequency in bog.most_common():
                word_list.append(word)
                values_list.append(frequency)
        else:
            for word, frequency in bog.items():
                word_list.append(word)
                values_list.append(frequency)
        X = np.asarray([values_list], dtype=np.float)
        # X.shape is a tuple of shape (num_elements_in_array, num_subelements_inside_each_element)
        features_length = X.shape[1]
        X_normalized = preprocessing.normalize(X, norm=norm)
        full = {word_list[i]: X_normalized[0, i] for i in range(features_length)}
        return full
