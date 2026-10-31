from rest_framework import serializers

from .models import Document


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = (
            'id', 'title', 'content', 'comments', 'version',
            'created_at', 'updated_at',
        )
        read_only_fields = ('id', 'created_at', 'updated_at')

    def validate_title(self, value):
        value = value.strip()
        return value or 'Untitled document'

    def validate_content(self, value):
        if not isinstance(value, dict) or value.get('type') != 'doc':
            raise serializers.ValidationError('Expected a Tiptap document object.')
        if not isinstance(value.get('content', []), list):
            raise serializers.ValidationError('Document content must be a list.')
        return value

    def validate_comments(self, value):
        if not isinstance(value, list) or any(not isinstance(comment, dict) for comment in value):
            raise serializers.ValidationError('Comments must be a list of objects.')
        return value

    def create(self, validated_data):
        validated_data.pop('version', None)
        return Document.objects.create(owner=self.context['request'].user, **validated_data)

    def update(self, instance, validated_data):
        validated_data.pop('version', None)
        for field, value in validated_data.items():
            setattr(instance, field, value)
        instance.version += 1
        instance.save()
        return instance


class DocumentListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ('id', 'title', 'version', 'created_at', 'updated_at')
