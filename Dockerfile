FROM google/deepvariant:0.7.2

WORKDIR /

RUN mkdir src
RUN unzip /opt/deepvariant/bin/call_variants.zip
RUN mv /runfiles /src

COPY lib/get_probs.py /src/runfiles/com_google_deepvariant/get_probs.py

WORKDIR /src/runfiles/com_google_deepvariant

ENTRYPOINT ["sh"]
